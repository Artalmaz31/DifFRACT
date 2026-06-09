import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from diffusers import FluxPipeline
from diffusers.models.embeddings import apply_rotary_emb


@dataclass
class LRMConfig:
    model_id: str = "black-forest-labs/FLUX.1-schnell"
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    d_model: int = 3072
    num_heads: int = 24
    head_dim: int = 128
    transcoder_dir: str = os.environ.get("TRANSCODERS_DIR", "transcoders")

    target_layers: Tuple[int, ...] = tuple(range(16))
    expansion_factor: int = 16
    time_embed_dim: int = 256

    height: int = 512
    width: int = 512
    num_inference_steps: int = 4
    guidance_scale: float = 0.0

    circuit_max_nodes: int = 1000
    expansion_batch_size: int = 50
    circuit_min_attribution: float = 1e-3

    prune_node_threshold_img: float = 0.8
    prune_node_threshold_txt: float = 0.8
    prune_edge_threshold_img: float = 0.98
    prune_edge_threshold_txt: float = 0.98

    @property
    def first_layer(self) -> int:
        return min(self.target_layers)

    @property
    def last_layer(self) -> int:
        return max(self.target_layers)

    @property
    def d_sparse(self) -> int:
        return self.d_model * self.expansion_factor


def apply_rope_flux(q: Tensor, k: Tensor, image_rotary_emb):
    if image_rotary_emb is None:
        return q, k

    q_rot = apply_rotary_emb(q, image_rotary_emb)
    k_rot = apply_rotary_emb(k, image_rotary_emb)
    return q_rot, k_rot


def compute_layernorm_inv_denom(x: Tensor, eps: float = 1e-6) -> Tensor:
    x = x.float()
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return torch.rsqrt(var + eps)


def run_temporal_aware_tc(tc, x_bsd: Tensor, timestep_b: Tensor):
    B, S, D = x_bsd.shape
    x_flat = x_bsd.reshape(B * S, D)
    t_flat = timestep_b.view(B, 1).expand(B, S).reshape(B * S)

    rec_flat, z_flat, h_pre_flat = tc.forward_with_preact(x_flat, t_flat)

    rec = rec_flat.reshape(B, S, D)
    z = z_flat.reshape(B, S, -1)
    h_pre = h_pre_flat.reshape(B, S, -1)

    return rec, z, h_pre


@dataclass
class AttentionCache:
    P: Tensor
    attn_error_img: Tensor
    attn_error_txt: Tensor
    S_txt: int
    S_img: int
    num_heads: int
    head_dim: int

    def get_P_on_device(self, device: torch.device) -> Tensor:
        return self.P.to(device=device, dtype=torch.float32, non_blocking=True)

    def get_errors_on_device(
        self, device: torch.device, dtype: torch.dtype
    ) -> Tuple[Tensor, Tensor]:
        return (
            self.attn_error_img.to(device=device, dtype=dtype, non_blocking=True),
            self.attn_error_txt.to(device=device, dtype=dtype, non_blocking=True),
        )


@dataclass
class LayerCache:
    layer_idx: int
    attention: Optional[AttentionCache] = None

    img_norm1_inv_denom: Optional[Tensor] = None
    txt_norm1_inv_denom: Optional[Tensor] = None
    img_norm2_inv_denom: Optional[Tensor] = None
    txt_norm2_inv_denom: Optional[Tensor] = None

    img_z: Optional[Tensor] = None
    txt_z: Optional[Tensor] = None
    img_h_pre: Optional[Tensor] = None
    txt_h_pre: Optional[Tensor] = None
    img_x_ff: Optional[Tensor] = None
    txt_x_ff: Optional[Tensor] = None

    img_ff_error: Optional[Tensor] = None
    txt_ff_error: Optional[Tensor] = None

    img_ff_out: Optional[Tensor] = None
    txt_ff_out: Optional[Tensor] = None

    img_gate_msa: Optional[Tensor] = None
    txt_gate_msa: Optional[Tensor] = None
    img_gate_mlp: Optional[Tensor] = None
    txt_gate_mlp: Optional[Tensor] = None
    img_scale_mlp: Optional[Tensor] = None
    txt_scale_mlp: Optional[Tensor] = None
    img_shift_mlp: Optional[Tensor] = None
    txt_shift_mlp: Optional[Tensor] = None


@dataclass
class FluxTrace:
    prompt: str = ""
    seed: int = 0
    step_idx: int = 0
    timestep: float = 0.0

    timestep_tensor: Optional[Tensor] = None
    layer_caches: Dict[int, LayerCache] = field(default_factory=dict)

    boundary_img: Optional[Tensor] = None
    boundary_txt: Optional[Tensor] = None

    transformer_kwargs: Dict[str, Any] = field(default_factory=dict)

    S_img: int = 0
    S_txt: int = 0

    def get_layer(self, idx: int) -> LayerCache:
        if idx not in self.layer_caches:
            self.layer_caches[idx] = LayerCache(layer_idx=idx)
        return self.layer_caches[idx]


def compute_flux_joint_attention(
    attn: nn.Module,
    hidden_states: Tensor,
    encoder_hidden_states: Tensor,
    image_rotary_emb,
) -> Dict[str, Tensor]:
    B = hidden_states.shape[0]
    S_img = hidden_states.shape[1]
    S_txt = encoder_hidden_states.shape[1]

    num_heads = attn.heads
    head_dim = getattr(attn, "head_dim", hidden_states.shape[-1] // num_heads)

    q_img = attn.to_q(hidden_states)
    k_img = attn.to_k(hidden_states)
    v_img = attn.to_v(hidden_states)

    q_txt = attn.add_q_proj(encoder_hidden_states)
    k_txt = attn.add_k_proj(encoder_hidden_states)
    v_txt = attn.add_v_proj(encoder_hidden_states)

    def reshape_heads(x: Tensor, seq_len: int) -> Tensor:
        return x.view(B, seq_len, num_heads, head_dim).transpose(1, 2)

    q_img = reshape_heads(q_img, S_img)
    k_img = reshape_heads(k_img, S_img)
    v_img = reshape_heads(v_img, S_img)

    q_txt = reshape_heads(q_txt, S_txt)
    k_txt = reshape_heads(k_txt, S_txt)
    v_txt = reshape_heads(v_txt, S_txt)

    q_img = attn.norm_q(q_img)
    k_img = attn.norm_k(k_img)
    q_txt = attn.norm_added_q(q_txt)
    k_txt = attn.norm_added_k(k_txt)

    q = torch.cat([q_txt, q_img], dim=2)
    k = torch.cat([k_txt, k_img], dim=2)
    v = torch.cat([v_txt, v_img], dim=2)

    q, k = apply_rope_flux(q, k, image_rotary_emb)

    scale = head_dim**-0.5

    q_f32 = q.float()
    k_f32 = k.float()

    scores = torch.matmul(q_f32, k_f32.transpose(-2, -1)) * scale

    scores_max = scores.amax(dim=-1, keepdim=True)
    scores = scores - scores_max

    probs = F.softmax(scores, dim=-1)
    out = torch.matmul(probs, v.float()).to(v.dtype)

    out_txt = out[:, :, :S_txt, :]
    out_img = out[:, :, S_txt:, :]

    out_txt = out_txt.transpose(1, 2).reshape(B, S_txt, -1)
    out_img = out_img.transpose(1, 2).reshape(B, S_img, -1)

    out_txt = attn.to_add_out(out_txt)
    out_img = attn.to_out[0](out_img)
    out_img = attn.to_out[1](out_img)

    return {
        "probs": probs,
        "out_img": out_img,
        "out_txt": out_txt,
        "S_txt": S_txt,
        "S_img": S_img,
        "num_heads": num_heads,
        "head_dim": head_dim,
    }


def capture_attention_with_errors(
    attn: nn.Module,
    hidden_states: Tensor,
    encoder_hidden_states: Tensor,
    image_rotary_emb,
    original_out_img: Tensor,
    original_out_txt: Tensor,
) -> AttentionCache:
    with torch.no_grad():
        result = compute_flux_joint_attention(
            attn, hidden_states, encoder_hidden_states, image_rotary_emb
        )

        attn_error_img = original_out_img - result["out_img"]
        attn_error_txt = original_out_txt - result["out_txt"]
        P_cpu = result["probs"].detach().cpu().to(torch.float32)

        return AttentionCache(
            P=P_cpu,
            attn_error_img=attn_error_img.detach().cpu().to(torch.float32),
            attn_error_txt=attn_error_txt.detach().cpu().to(torch.float32),
            S_txt=result["S_txt"],
            S_img=result["S_img"],
            num_heads=result["num_heads"],
            head_dim=result["head_dim"],
        )


class FrozenNormWrapper(nn.Module):
    def __init__(self, original_norm: nn.Module, inv_denom: Tensor):
        super().__init__()
        self.register_buffer("inv_denom", inv_denom.detach(), persistent=False)

    def forward(self, x):
        out_dtype = x.dtype
        x = x.float()
        inv = self.inv_denom.float()
        mean = x.mean(dim=-1, keepdim=True)
        x_norm = (x - mean) * inv
        return x_norm.to(out_dtype)


class LRMFFWrapper(nn.Module):
    def __init__(
        self,
        transcoder: nn.Module = None,
        error_term: Tensor = None,
        timestep: Tensor = None,
        time_embed_dim: int = 256,
        linear_mode: bool = False,
        cached_output: Tensor = None,
    ):
        super().__init__()

        self.transcoder = transcoder
        self.time_embed_dim = time_embed_dim
        self.linear_mode = linear_mode

        if error_term is not None:
            self.register_buffer("error_term", error_term.detach(), persistent=False)
        else:
            self.error_term = None
        if timestep is not None:
            self.register_buffer(
                "timestep", timestep.detach().view(-1), persistent=False
            )
        else:
            self.timestep = None
        if cached_output is not None:
            self.register_buffer(
                "cached_output", cached_output.detach(), persistent=False
            )
        else:
            self.cached_output = None

        self.last_z: Optional[Tensor] = None
        self.last_h_pre: Optional[Tensor] = None
        self.last_x_ff: Optional[Tensor] = None
        self.ablation_specs: List[Tuple[int, int]] = []

    def forward(self, x: Tensor) -> Tensor:
        if self.linear_mode and self.cached_output is not None:
            return self.cached_output.to(x.dtype)

        tc_on_cpu = next(self.transcoder.parameters()).device.type == "cpu"
        if tc_on_cpu:
            self.transcoder.to(x.device)

        B, S, D = x.shape
        tc_dtype = next(self.transcoder.parameters()).dtype

        x_tc = x.to(dtype=tc_dtype)
        t_b = self.timestep.to(device=x.device, dtype=torch.float32).view(-1)
        rec, z, h_pre = run_temporal_aware_tc(self.transcoder, x_tc, t_b)

        if self.ablation_specs:
            z = z.clone()
            for pos, feat_idx in self.ablation_specs:
                z[:, pos, feat_idx] = 0.0

        rec = F.linear(
            z, self.transcoder.decoder.weight, self.transcoder.decoder.bias
        ).float()

        self.last_z = z.detach()
        self.last_h_pre = h_pre.detach()
        self.last_x_ff = x.detach()

        if self.linear_mode:
            rec = rec.detach()

        y = rec + self.error_term.to(device=x.device, dtype=torch.float32)

        if tc_on_cpu:
            self.transcoder.cpu()

        return y.to(x.dtype)


class FrozenAttentionWrapper(nn.Module):
    def __init__(
        self,
        original_attn: nn.Module,
        cache: AttentionCache,
    ):
        super().__init__()
        self.original_attn = original_attn
        self.cache = cache

        self.to_v = original_attn.to_v
        self.add_v_proj = original_attn.add_v_proj
        self.to_out = original_attn.to_out
        self.to_add_out = original_attn.to_add_out

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        image_rotary_emb=None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        device = hidden_states.device
        dtype = hidden_states.dtype
        B = hidden_states.shape[0]
        S_img = self.cache.S_img
        S_txt = self.cache.S_txt
        num_heads = self.cache.num_heads
        head_dim = self.cache.head_dim

        v_img = self.to_v(hidden_states)
        v_txt = self.add_v_proj(encoder_hidden_states)

        v_img = v_img.view(B, S_img, num_heads, head_dim).transpose(1, 2)
        v_txt = v_txt.view(B, S_txt, num_heads, head_dim).transpose(1, 2)

        V = torch.cat([v_txt, v_img], dim=2)
        P = self.cache.get_P_on_device(device)

        V_f32 = V.float()
        out = torch.matmul(P, V_f32).to(dtype)

        out_txt = out[:, :, :S_txt, :]
        out_img = out[:, :, S_txt:, :]

        out_txt = out_txt.transpose(1, 2).reshape(B, S_txt, -1)
        out_img = out_img.transpose(1, 2).reshape(B, S_img, -1)

        out_txt = self.to_add_out(out_txt)
        out_img = self.to_out[0](out_img)
        out_img = self.to_out[1](out_img)

        err_img, err_txt = self.cache.get_errors_on_device(device, dtype)
        return out_img + err_img, out_txt + err_txt


class FluxTraceCapturer:
    def __init__(
        self,
        pipe: "FluxPipeline",
        transcoders: Dict[str, nn.Module],
        cfg: "LRMConfig",
    ):
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.transcoders = transcoders
        self.cfg = cfg

        self.hook_handles: List[Any] = []
        self.restore_fns: List[Callable] = []

    def _clear_hooks(self):
        for h in self.hook_handles:
            try:
                h.remove()
            except:
                pass
        self.hook_handles.clear()

        for fn in self.restore_fns:
            try:
                fn()
            except:
                pass
        self.restore_fns.clear()

    @torch.no_grad()
    def capture(
        self,
        prompt: str,
        seed: int,
        target_step: int = 0,
    ) -> FluxTrace:
        self._clear_hooks()
        trace = FluxTrace(prompt=prompt, seed=seed, step_idx=target_step)

        step_state = {
            "transformer_call_idx": -1,
            "captured": False,
            "timestep": None,
        }

        def transformer_pre_hook(module, args, kwargs):
            step_state["transformer_call_idx"] += 1
            current_step = step_state["transformer_call_idx"]

            if current_step == target_step and not step_state["captured"]:
                trace.transformer_kwargs = {
                    k: (v.detach().clone() if isinstance(v, Tensor) else v)
                    for k, v in kwargs.items()
                }

                timestep = kwargs.get("timestep")
                if timestep is not None:
                    step_state["timestep"] = timestep.detach().clone()
                    trace.timestep_tensor = timestep.detach().cpu()
                    trace.timestep = float(timestep.mean().item())

        self.hook_handles.append(
            self.transformer.register_forward_pre_hook(
                transformer_pre_hook, with_kwargs=True
            )
        )

        def make_block_pre_hook(layer_idx: int):
            def hook(module, args, kwargs):
                if step_state["captured"]:
                    return
                if step_state["transformer_call_idx"] != target_step:
                    return

                hs = kwargs.get("hidden_states", args[0] if args else None)
                ehs = kwargs.get(
                    "encoder_hidden_states", args[1] if len(args) > 1 else None
                )

                if layer_idx == self.cfg.first_layer:
                    trace.boundary_img = hs.detach().cpu()
                    trace.boundary_txt = ehs.detach().cpu()
                    trace.S_img = hs.shape[1]
                    trace.S_txt = ehs.shape[1]

            return hook

        def make_adaln_hook(layer_idx: int, stream: str):
            def hook(module, args, output):
                if step_state["captured"]:
                    return
                if step_state["transformer_call_idx"] != target_step:
                    return

                lc = trace.get_layer(layer_idx)

                if isinstance(output, tuple) and len(output) >= 5:
                    normalized, gate_msa, shift_mlp, scale_mlp, gate_mlp = output[:5]

                    if stream == "img":
                        lc.img_gate_msa = gate_msa.detach().cpu().to(torch.float32)
                        lc.img_gate_mlp = gate_mlp.detach().cpu().to(torch.float32)
                        lc.img_scale_mlp = scale_mlp.detach().cpu().to(torch.float32)
                        lc.img_shift_mlp = shift_mlp.detach().cpu().to(torch.float32)
                    else:
                        lc.txt_gate_msa = gate_msa.detach().cpu().to(torch.float32)
                        lc.txt_gate_mlp = gate_mlp.detach().cpu().to(torch.float32)
                        lc.txt_scale_mlp = scale_mlp.detach().cpu().to(torch.float32)
                        lc.txt_shift_mlp = shift_mlp.detach().cpu().to(torch.float32)

                    x = args[0]
                    inner_norm = module.norm
                    eps = getattr(inner_norm, "eps", 1e-6)
                    inv_denom = compute_layernorm_inv_denom(x, eps)

                    if stream == "img":
                        lc.img_norm1_inv_denom = (
                            inv_denom.detach().cpu().to(torch.float32)
                        )
                    else:
                        lc.txt_norm1_inv_denom = (
                            inv_denom.detach().cpu().to(torch.float32)
                        )

            return hook

        def make_norm2_hook(layer_idx: int, stream: str):
            def hook(module, args, output):
                if step_state["captured"]:
                    return
                if step_state["transformer_call_idx"] != target_step:
                    return

                x = args[0]
                eps = getattr(module, "eps", 1e-6)
                inv_denom = compute_layernorm_inv_denom(x, eps)

                lc = trace.get_layer(layer_idx)
                if stream == "img":
                    lc.img_norm2_inv_denom = inv_denom.detach().cpu().to(torch.float32)
                else:
                    lc.txt_norm2_inv_denom = inv_denom.detach().cpu().to(torch.float32)

            return hook

        def make_ff_hook(layer_idx: int, stream: str):
            def hook(module, args, output):
                if step_state["captured"]:
                    return
                if step_state["transformer_call_idx"] != target_step:
                    return

                x = args[0]
                y_true = output.float()

                tc_key = f"{stream}_{layer_idx}"
                tc = self.transcoders.get(tc_key)

                if tc is None:
                    return

                timestep = step_state.get("timestep")
                if timestep is None:
                    return

                tc.to(x.device)

                t_b = timestep.to(device=x.device, dtype=torch.float32).view(-1)
                tc_dtype = next(tc.parameters()).dtype
                rec, z, h_pre = run_temporal_aware_tc(tc, x.to(tc_dtype), t_b)

                rec_full = F.linear(z, tc.decoder.weight, tc.decoder.bias)
                error = y_true.detach().float() - rec_full.detach().float()

                tc.cpu()

                lc = trace.get_layer(layer_idx)
                if stream == "img":
                    lc.img_x_ff = x.detach().cpu().to(torch.float32)
                    lc.img_z = z.detach().cpu().to(torch.float32)
                    lc.img_h_pre = h_pre.detach().cpu().to(torch.float32)
                    lc.img_ff_error = error.detach().cpu().to(torch.float32)
                    lc.img_ff_out = y_true.detach().cpu()
                else:
                    lc.txt_x_ff = x.detach().cpu().to(torch.float32)
                    lc.txt_z = z.detach().cpu().to(torch.float32)
                    lc.txt_h_pre = h_pre.detach().cpu().to(torch.float32)
                    lc.txt_ff_error = error.detach().cpu().to(torch.float32)
                    lc.txt_ff_out = y_true.detach().cpu()

            return hook

        def make_attn_wrapper(layer_idx: int, original_forward):
            def wrapped_forward(
                hidden_states,
                encoder_hidden_states=None,
                image_rotary_emb=None,
                **kwargs,
            ):

                if (
                    step_state["captured"]
                    or step_state["transformer_call_idx"] != target_step
                ):
                    return original_forward(
                        hidden_states,
                        encoder_hidden_states,
                        image_rotary_emb=image_rotary_emb,
                        **kwargs,
                    )

                out_img, out_txt = original_forward(
                    hidden_states,
                    encoder_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    **kwargs,
                )

                blk = self.transformer.transformer_blocks[layer_idx]
                attn_cache = capture_attention_with_errors(
                    attn=blk.attn,
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    original_out_img=out_img,
                    original_out_txt=out_txt,
                )

                trace.get_layer(layer_idx).attention = attn_cache

                return out_img, out_txt

            return wrapped_forward

        for layer_idx in self.cfg.target_layers:
            blk = self.transformer.transformer_blocks[layer_idx]

            self.hook_handles.append(
                blk.register_forward_pre_hook(
                    make_block_pre_hook(layer_idx), with_kwargs=True
                )
            )

            self.hook_handles.append(
                blk.norm1.register_forward_hook(make_adaln_hook(layer_idx, "img"))
            )
            self.hook_handles.append(
                blk.norm1_context.register_forward_hook(
                    make_adaln_hook(layer_idx, "txt")
                )
            )

            self.hook_handles.append(
                blk.norm2.register_forward_hook(make_norm2_hook(layer_idx, "img"))
            )
            self.hook_handles.append(
                blk.norm2_context.register_forward_hook(
                    make_norm2_hook(layer_idx, "txt")
                )
            )

            self.hook_handles.append(
                blk.ff.register_forward_hook(make_ff_hook(layer_idx, "img"))
            )
            self.hook_handles.append(
                blk.ff_context.register_forward_hook(make_ff_hook(layer_idx, "txt"))
            )

            original_forward = blk.attn.forward
            blk.attn.forward = make_attn_wrapper(layer_idx, original_forward)
            self.restore_fns.append(
                lambda blk=blk, orig=original_forward: setattr(
                    blk.attn, "forward", orig
                )
            )

        generator = torch.Generator(device=self.cfg.device).manual_seed(seed)

        def step_callback(pipe, step_idx, timestep, callback_kwargs):
            if step_state["transformer_call_idx"] >= target_step:
                step_state["captured"] = True
            return callback_kwargs

        _ = self.pipe(
            prompt,
            prompt_2=prompt,
            height=self.cfg.height,
            width=self.cfg.width,
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
            generator=generator,
            callback_on_step_end=step_callback,
            output_type="latent",
        )

        self._clear_hooks()
        return trace


class LRMPatcher:
    def __init__(
        self,
        transformer: nn.Module,
        trace: FluxTrace,
        transcoders: Dict[str, nn.Module],
        cfg: "LRMConfig",
        mode: str = "linear",
    ):
        self.transformer = transformer
        self.trace = trace
        self.transcoders = transcoders
        self.cfg = cfg
        self.mode = mode

        self._originals: Dict[int, Dict[str, Any]] = {}
        self._patched = False

    def _patch_layer(self, layer_idx: int):
        blk = self.transformer.transformer_blocks[layer_idx]
        lc = self.trace.get_layer(layer_idx)

        self._originals[layer_idx] = {
            "ff": blk.ff,
            "ff_context": blk.ff_context,
            "norm2": blk.norm2,
            "norm2_context": blk.norm2_context,
            "norm1_norm": blk.norm1.norm,
            "norm1_context_norm": blk.norm1_context.norm,
            "attn_forward": blk.attn.forward,
        }

        device = self.cfg.device
        linear_mode = self.mode == "linear"

        if linear_mode and lc.img_ff_out is not None:
            blk.ff = LRMFFWrapper(
                cached_output=lc.img_ff_out.to(device),
                linear_mode=True,
            )
        else:
            tc_img = self.transcoders.get(f"img_{layer_idx}")
            if tc_img is not None and lc.img_ff_error is not None:
                blk.ff = LRMFFWrapper(
                    transcoder=tc_img,
                    error_term=lc.img_ff_error.to(device),
                    timestep=self.trace.timestep_tensor.to(device),
                    time_embed_dim=self.cfg.time_embed_dim,
                    linear_mode=linear_mode,
                )

        if linear_mode and lc.txt_ff_out is not None:
            blk.ff_context = LRMFFWrapper(
                cached_output=lc.txt_ff_out.to(device),
                linear_mode=True,
            )
        else:
            tc_txt = self.transcoders.get(f"txt_{layer_idx}")
            if tc_txt is not None and lc.txt_ff_error is not None:
                blk.ff_context = LRMFFWrapper(
                    transcoder=tc_txt,
                    error_term=lc.txt_ff_error.to(device),
                    timestep=self.trace.timestep_tensor.to(device),
                    time_embed_dim=self.cfg.time_embed_dim,
                    linear_mode=linear_mode,
                )

        if lc.img_norm2_inv_denom is not None:
            blk.norm2 = FrozenNormWrapper(
                blk.norm2,
                lc.img_norm2_inv_denom.to(device),
            )

        if lc.txt_norm2_inv_denom is not None:
            blk.norm2_context = FrozenNormWrapper(
                blk.norm2_context,
                lc.txt_norm2_inv_denom.to(device),
            )

        if lc.img_norm1_inv_denom is not None:
            blk.norm1.norm = FrozenNormWrapper(
                blk.norm1.norm,
                lc.img_norm1_inv_denom.to(device),
            )

        if lc.txt_norm1_inv_denom is not None:
            blk.norm1_context.norm = FrozenNormWrapper(
                blk.norm1_context.norm,
                lc.txt_norm1_inv_denom.to(device),
            )

        if lc.attention is not None:
            wrapper = FrozenAttentionWrapper(
                original_attn=blk.attn,
                cache=lc.attention,
            )

            def make_wrapped_forward(w):
                def wrapped(
                    hidden_states,
                    encoder_hidden_states=None,
                    image_rotary_emb=None,
                    **kwargs,
                ):
                    return w(
                        hidden_states,
                        encoder_hidden_states,
                        image_rotary_emb=image_rotary_emb,
                        **kwargs,
                    )

                return wrapped

            blk.attn.forward = make_wrapped_forward(wrapper)

    def _restore_layer(self, layer_idx: int):
        if layer_idx not in self._originals:
            return

        orig = self._originals[layer_idx]
        blk = self.transformer.transformer_blocks[layer_idx]

        blk.ff = orig["ff"]
        blk.ff_context = orig["ff_context"]
        blk.norm2 = orig["norm2"]
        blk.norm2_context = orig["norm2_context"]
        blk.norm1.norm = orig["norm1_norm"]
        blk.norm1_context.norm = orig["norm1_context_norm"]
        blk.attn.forward = orig["attn_forward"]

    def __enter__(self):
        for layer_idx in self.cfg.target_layers:
            self._patch_layer(layer_idx)
        self._patched = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for layer_idx in self.cfg.target_layers:
            self._restore_layer(layer_idx)
        self._originals.clear()
        self._patched = False
        torch.cuda.empty_cache()
        return False
