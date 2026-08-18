from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from diffusers.models.embeddings import apply_rotary_emb
from transcoder_training.arch import ARCHS, runs_real_cfg as _training_runs_real_cfg


@dataclass(frozen=True)
class TracingSpec:
    name: str
    family: str  # "flux" | "sd3"
    pipeline_cls: str
    default_model_id: str
    d_model: int
    num_heads: int
    head_dim: int
    num_layers: int
    num_inference_steps: int
    guidance_scale: float
    timestep_scale: float = 1.0
    prompt_aliases: Tuple[str, ...] = ("prompt_2",)


def _tracing_spec(name: str, **kwargs) -> TracingSpec:
    train = ARCHS[name]
    return TracingSpec(
        name=train.name,
        pipeline_cls=train.pipeline_cls,
        default_model_id=train.default_model_id,
        d_model=train.d_model,
        num_inference_steps=train.num_inference_steps,
        guidance_scale=train.guidance_scale,
        timestep_scale=train.timestep_scale,
        prompt_aliases=train.prompt_aliases,
        **kwargs,
    )


FLUX_SCHNELL = _tracing_spec(
    "flux-schnell", family="flux", num_heads=24, head_dim=128, num_layers=19
)
FLUX_DEV = _tracing_spec(
    "flux-dev", family="flux", num_heads=24, head_dim=128, num_layers=19
)
SD3_MEDIUM = _tracing_spec(
    "sd3-medium", family="sd3", num_heads=24, head_dim=64, num_layers=24
)
SD35_MEDIUM = _tracing_spec(
    "sd3.5-medium", family="sd3", num_heads=24, head_dim=64, num_layers=24
)

TRACING_ARCHS = {
    spec.name: spec for spec in (FLUX_SCHNELL, FLUX_DEV, SD3_MEDIUM, SD35_MEDIUM)
}


def _heads(x: Tensor, batch: int, seq: int, num_heads: int, head_dim: int) -> Tensor:
    """(B, S, H*D) -> (B, H, S, D)"""
    return x.view(batch, seq, num_heads, head_dim).transpose(1, 2)


def _unheads(x: Tensor, batch: int, seq: int) -> Tensor:
    """(B, H, S, D) -> (B, S, H*D)"""
    return x.transpose(1, 2).reshape(batch, seq, -1)


def _softmax_probs(q: Tensor, k: Tensor, scale: float) -> Tensor:
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    scores = scores - scores.amax(dim=-1, keepdim=True)
    return F.softmax(scores, dim=-1)


class Backend:
    """Base adapter. Subclasses only override what actually differs."""

    family: str = ""
    joint_order: Tuple[str, str] = ("txt", "img")
    uses_rope: bool = False
    attn_extra_keys: Tuple[str, ...] = ()

    def __init__(self, spec: TracingSpec):
        self.spec = spec

    @staticmethod
    def blocks(transformer: nn.Module) -> nn.ModuleList:
        return transformer.transformer_blocks

    @staticmethod
    def ff(block: nn.Module, stream: str) -> Optional[nn.Module]:
        return block.ff if stream == "img" else getattr(block, "ff_context", None)

    @staticmethod
    def norm2(block: nn.Module, stream: str) -> Optional[nn.Module]:
        return block.norm2 if stream == "img" else getattr(block, "norm2_context", None)

    @staticmethod
    def norm1(block: nn.Module, stream: str) -> Optional[nn.Module]:
        return block.norm1 if stream == "img" else getattr(block, "norm1_context", None)

    @staticmethod
    def dual_attn(block: nn.Module) -> Optional[nn.Module]:
        return getattr(block, "attn2", None)

    def streams(self, block: nn.Module) -> Tuple[str, ...]:
        return ("img", "txt") if self.ff(block, "txt") is not None else ("img",)

    def head_shape(self, attn: nn.Module) -> Tuple[int, int]:
        num_heads = attn.heads
        return num_heads, attn.to_q.out_features // num_heads

    def cat_joint(self, img: Tensor, txt: Tensor, dim: int) -> Tensor:
        first, second = (txt, img) if self.joint_order[0] == "txt" else (img, txt)
        return torch.cat([first, second], dim=dim)

    def split_joint(
        self, joint: Tensor, s_img: int, s_txt: int, dim: int
    ) -> Tuple[Tensor, Tensor]:
        first_len = s_txt if self.joint_order[0] == "txt" else s_img
        first, second = joint.split_with_sizes(
            [first_len, joint.shape[dim] - first_len], dim=dim
        )
        return (second, first) if self.joint_order[0] == "txt" else (first, second)

    def apply_rope(
        self, q: Tensor, k: Tensor, image_rotary_emb
    ) -> Tuple[Tensor, Tensor]:
        if not self.uses_rope or image_rotary_emb is None:
            return q, k
        return apply_rotary_emb(q, image_rotary_emb), apply_rotary_emb(
            k, image_rotary_emb
        )

    def joint_attention(
        self,
        attn: nn.Module,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        image_rotary_emb=None,
    ) -> Dict[str, Any]:
        """Recompute joint attention."""
        B, S_img, _ = hidden_states.shape
        S_txt = encoder_hidden_states.shape[1]
        num_heads, head_dim = self.head_shape(attn)

        q_img = _heads(attn.to_q(hidden_states), B, S_img, num_heads, head_dim)
        k_img = _heads(attn.to_k(hidden_states), B, S_img, num_heads, head_dim)
        v_img = _heads(attn.to_v(hidden_states), B, S_img, num_heads, head_dim)

        q_txt = _heads(
            attn.add_q_proj(encoder_hidden_states), B, S_txt, num_heads, head_dim
        )
        k_txt = _heads(
            attn.add_k_proj(encoder_hidden_states), B, S_txt, num_heads, head_dim
        )
        v_txt = _heads(
            attn.add_v_proj(encoder_hidden_states), B, S_txt, num_heads, head_dim
        )

        if attn.norm_q is not None:
            q_img = attn.norm_q(q_img)
        if attn.norm_k is not None:
            k_img = attn.norm_k(k_img)
        if getattr(attn, "norm_added_q", None) is not None:
            q_txt = attn.norm_added_q(q_txt)
        if getattr(attn, "norm_added_k", None) is not None:
            k_txt = attn.norm_added_k(k_txt)

        q = self.cat_joint(q_img, q_txt, dim=2)
        k = self.cat_joint(k_img, k_txt, dim=2)
        v = self.cat_joint(v_img, v_txt, dim=2)

        q, k = self.apply_rope(q, k, image_rotary_emb)

        probs = _softmax_probs(q, k, head_dim**-0.5)
        out = torch.matmul(probs, v.float()).to(v.dtype)

        out_img, out_txt = self.split_joint(out, S_img, S_txt, dim=2)
        out_img = self.project_out_img(attn, _unheads(out_img, B, S_img))
        out_txt = self.project_out_txt(attn, _unheads(out_txt, B, S_txt))

        return {
            "probs": probs,
            "out_img": out_img,
            "out_txt": out_txt,
            "S_txt": S_txt,
            "S_img": S_img,
            "num_heads": num_heads,
            "head_dim": head_dim,
        }

    def self_attention(self, attn: nn.Module, hidden_states: Tensor) -> Dict[str, Any]:
        """Recompute an image-only self-attention (SD3.5)."""
        B, S, _ = hidden_states.shape
        num_heads, head_dim = self.head_shape(attn)

        q = _heads(attn.to_q(hidden_states), B, S, num_heads, head_dim)
        k = _heads(attn.to_k(hidden_states), B, S, num_heads, head_dim)
        v = _heads(attn.to_v(hidden_states), B, S, num_heads, head_dim)

        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)

        probs = _softmax_probs(q, k, head_dim**-0.5)
        out = torch.matmul(probs, v.float()).to(v.dtype)
        out = self.project_out_img(attn, _unheads(out, B, S))

        return {
            "probs": probs,
            "out_img": out,
            "S_img": S,
            "num_heads": num_heads,
            "head_dim": head_dim,
        }

    @staticmethod
    def project_out_img(attn: nn.Module, x: Tensor) -> Tensor:
        x = attn.to_out[0](x)
        return attn.to_out[1](x)

    @staticmethod
    def project_out_txt(attn: nn.Module, x: Tensor) -> Tensor:
        if getattr(attn, "to_add_out", None) is None:
            return x
        return attn.to_add_out(x)

    def attn_extra_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {k: kwargs[k] for k in self.attn_extra_keys if k in kwargs}

    def runs_real_cfg(self, guidance_scale: float) -> bool:
        """True when the pipeline stacks [uncond, cond] and so doubles the transformer batch."""
        return _training_runs_real_cfg(self.spec.pipeline_cls, guidance_scale)

    def generation_kwargs(self, prompt, guidance_scale: float) -> Dict[str, Any]:
        kwargs = {alias: prompt for alias in self.spec.prompt_aliases}
        kwargs["guidance_scale"] = guidance_scale
        return kwargs

    def image_grid(self, height: int, width: int) -> Tuple[int, int]:
        return height // 16, width // 16

    def decode_latents(
        self, pipe, latents: Tensor, height: int, width: int
    ) -> np.ndarray:
        raise NotImplementedError

    def tokenize_for_display(self, pipe, prompt: str) -> List[str]:
        raise NotImplementedError

    @staticmethod
    def _vae_decode(pipe, lat: Tensor) -> np.ndarray:
        cfg = pipe.vae.config
        lat = lat / cfg.scaling_factor + getattr(cfg, "shift_factor", 0.0)
        image = pipe.vae.decode(lat, return_dict=False)[0]
        image = (image.float().cpu() / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1).numpy()
        return (image * 255).round().astype(np.uint8)[0]


class FluxBackend(Backend):
    family = "flux"
    joint_order = ("txt", "img")
    uses_rope = True
    attn_extra_keys = ("image_rotary_emb",)

    def decode_latents(
        self, pipe, latents: Tensor, height: int, width: int
    ) -> np.ndarray:
        with torch.no_grad():
            lat = latents.to(device=pipe.vae.device, dtype=pipe.vae.dtype)

            batch_size = lat.shape[0]
            h_latent, w_latent = 2 * (height // 16), 2 * (width // 16)
            channels = lat.shape[-1]

            lat = lat.view(
                batch_size, h_latent // 2, w_latent // 2, channels // 4, 2, 2
            )
            lat = lat.permute(0, 3, 1, 4, 2, 5)
            lat = lat.reshape(batch_size, channels // 4, h_latent, w_latent)

            return self._vae_decode(pipe, lat)

    def tokenize_for_display(self, pipe, prompt: str) -> List[str]:
        enc = pipe.tokenizer_2(
            prompt, truncation=True, max_length=512, return_tensors="pt"
        )
        return pipe.tokenizer_2.convert_ids_to_tokens(enc.input_ids[0].tolist())


class SD3Backend(Backend):
    family = "sd3"
    joint_order = ("img", "txt")
    uses_rope = False
    attn_extra_keys = ()

    def decode_latents(
        self, pipe, latents: Tensor, height: int, width: int
    ) -> np.ndarray:
        with torch.no_grad():
            lat = latents.to(device=pipe.vae.device, dtype=pipe.vae.dtype)
            return self._vae_decode(pipe, lat)

    def tokenize_for_display(self, pipe, prompt: str) -> List[str]:
        clip = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        clip_toks = pipe.tokenizer.convert_ids_to_tokens(clip.input_ids[0].tolist())

        t5 = pipe.tokenizer_3(
            prompt, truncation=True, max_length=256, return_tensors="pt"
        )
        t5_toks = pipe.tokenizer_3.convert_ids_to_tokens(t5.input_ids[0].tolist())

        return clip_toks + t5_toks


def get_spec(name: str) -> TracingSpec:
    spec = TRACING_ARCHS.get(name)
    if spec is None:
        raise ValueError(
            f"Unknown backbone {name!r}. Known: {', '.join(sorted(TRACING_ARCHS))}"
        )
    return spec


def get_backend(name: str) -> Backend:
    spec = get_spec(name)
    if spec.family == "flux":
        return FluxBackend(spec)
    if spec.family == "sd3":
        return SD3Backend(spec)
    raise ValueError(f"Unknown backend family {spec.family!r} for {name!r}")
