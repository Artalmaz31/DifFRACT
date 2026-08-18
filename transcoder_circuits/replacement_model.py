import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
import diffusers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .backends import (
    Backend,
    TracingSpec,
    get_backend,
    get_spec,
)


@dataclass
class LRMConfig:
    backend: str = "flux-schnell"
    pipeline_cls: str = "FluxPipeline"
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
    timestep_scale: float = 1.0
    prompt_aliases: Tuple[str, ...] = ("prompt_2",)
    cfg_branch: str = "cond"

    circuit_max_nodes: int = 1000
    expansion_batch_size: int = 50
    circuit_min_attribution: float = 1e-3

    prune_node_threshold_img: float = 0.8
    prune_node_threshold_txt: float = 0.8
    prune_edge_threshold_img: float = 0.98
    prune_edge_threshold_txt: float = 0.98

    @classmethod
    def for_model(cls, name: str, **overrides) -> "LRMConfig":
        spec = get_spec(name)
        base: Dict[str, Any] = dict(
            backend=spec.name,
            pipeline_cls=spec.pipeline_cls,
            model_id=spec.default_model_id,
            d_model=spec.d_model,
            num_heads=spec.num_heads,
            head_dim=spec.head_dim,
            num_inference_steps=spec.num_inference_steps,
            guidance_scale=spec.guidance_scale,
            timestep_scale=spec.timestep_scale,
            prompt_aliases=spec.prompt_aliases,
        )
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**base)

    @property
    def spec(self) -> TracingSpec:
        return get_spec(self.backend)

    def get_backend(self) -> Backend:
        return get_backend(self.backend)

    @property
    def runs_real_cfg(self) -> bool:
        return self.get_backend().runs_real_cfg(self.guidance_scale)

    @property
    def first_layer(self) -> int:
        return min(self.target_layers)

    @property
    def last_layer(self) -> int:
        return max(self.target_layers)

    @property
    def d_sparse(self) -> int:
        return self.d_model * self.expansion_factor

    def generation_kwargs(self, prompt) -> Dict[str, Any]:
        return self.get_backend().generation_kwargs(prompt, self.guidance_scale)


def compute_layernorm_inv_denom(x: Tensor, eps: float = 1e-6) -> Tensor:
    x = x.float()
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return torch.rsqrt(var + eps)


def traced_rows(batch: int, cached: int, branch: Optional[str]) -> Optional[slice]:
    if batch == cached:
        return None
    if branch is not None and batch == 2 * cached:
        return slice(cached, batch) if branch == "cond" else slice(0, cached)
    raise ValueError(
        f"the LRM was traced on a batch of {cached} "
        f"but is being run on a batch of {batch}."
    )


def _splice(original: Tensor, replacement: Tensor, rows: slice) -> Tensor:
    out = original.clone()
    out[rows] = replacement.to(dtype=out.dtype)
    return out


def run_temporal_aware_tc(tc, x_bsd: Tensor, timestep_b: Tensor):
    B, S, D = x_bsd.shape
    x_flat = x_bsd.reshape(B * S, D)
    t = timestep_b.reshape(-1)
    if t.numel() == 1 and B > 1:
        t = t.expand(B)
    t_flat = t.view(B, 1).expand(B, S).reshape(B * S)

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
class SelfAttentionCache:
    P: Tensor
    attn_error_img: Tensor
    S_img: int
    num_heads: int
    head_dim: int

    def get_P_on_device(self, device: torch.device) -> Tensor:
        return self.P.to(device=device, dtype=torch.float32, non_blocking=True)

    def get_error_on_device(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        return self.attn_error_img.to(device=device, dtype=dtype, non_blocking=True)


@dataclass
class LayerCache:
    layer_idx: int
    attention: Optional[AttentionCache] = None
    attention2: Optional[SelfAttentionCache] = None

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
    img_gate_msa2: Optional[Tensor] = None
    img_shift_msa: Optional[Tensor] = None
    txt_shift_msa: Optional[Tensor] = None
    img_shift_msa2: Optional[Tensor] = None
    img_gate_mlp: Optional[Tensor] = None
    txt_gate_mlp: Optional[Tensor] = None
    img_scale_mlp: Optional[Tensor] = None
    txt_scale_mlp: Optional[Tensor] = None
    img_shift_mlp: Optional[Tensor] = None
    txt_shift_mlp: Optional[Tensor] = None


@dataclass
class Trace:
    prompt: str = ""
    seed: int = 0
    step_idx: int = 0
    timestep: float = 0.0
    timestep_raw: float = 0.0

    timestep_tensor: Optional[Tensor] = None
    layer_caches: Dict[int, LayerCache] = field(default_factory=dict)

    boundary_img: Optional[Tensor] = None
    boundary_txt: Optional[Tensor] = None

    transformer_kwargs: Dict[str, Any] = field(default_factory=dict)

    S_img: int = 0
    S_txt: int = 0

    backend: str = "flux-schnell"
    cfg_branch: Optional[str] = None

    def get_layer(self, idx: int) -> LayerCache:
        if idx not in self.layer_caches:
            self.layer_caches[idx] = LayerCache(layer_idx=idx)
        return self.layer_caches[idx]


def capture_attention_with_errors(
    backend: Backend,
    attn: nn.Module,
    hidden_states: Tensor,
    encoder_hidden_states: Tensor,
    original_out_img: Tensor,
    original_out_txt: Tensor,
    select=lambda x: x,
    **extra,
) -> AttentionCache:
    with torch.no_grad():
        result = backend.joint_attention(
            attn, hidden_states, encoder_hidden_states, **extra
        )

        attn_error_img = original_out_img - result["out_img"]
        attn_error_txt = original_out_txt - result["out_txt"]

        return AttentionCache(
            P=select(result["probs"]).detach().cpu().to(torch.float32),
            attn_error_img=select(attn_error_img).detach().cpu().to(torch.float32),
            attn_error_txt=select(attn_error_txt).detach().cpu().to(torch.float32),
            S_txt=result["S_txt"],
            S_img=result["S_img"],
            num_heads=result["num_heads"],
            head_dim=result["head_dim"],
        )


def capture_self_attention_with_errors(
    backend: Backend,
    attn: nn.Module,
    hidden_states: Tensor,
    original_out: Tensor,
    select=lambda x: x,
) -> SelfAttentionCache:
    with torch.no_grad():
        result = backend.self_attention(attn, hidden_states)
        error = original_out - result["out_img"]

        return SelfAttentionCache(
            P=select(result["probs"]).detach().cpu().to(torch.float32),
            attn_error_img=select(error).detach().cpu().to(torch.float32),
            S_img=result["S_img"],
            num_heads=result["num_heads"],
            head_dim=result["head_dim"],
        )


class FrozenNormWrapper(nn.Module):
    def __init__(
        self,
        original_norm: nn.Module,
        inv_denom: Tensor,
        branch: Optional[str] = None,
    ):
        super().__init__()
        self.original_norm = original_norm
        self.branch = branch
        self.register_buffer("inv_denom", inv_denom.detach(), persistent=False)

    def _frozen(self, x: Tensor, inv: Tensor) -> Tensor:
        xf = x.float()
        mean = xf.mean(dim=-1, keepdim=True)
        return ((xf - mean) * inv).to(x.dtype)

    def forward(self, x):
        inv = self.inv_denom.float()
        rows = traced_rows(x.shape[0], inv.shape[0], self.branch)
        if rows is not None:
            return _splice(self.original_norm(x), self._frozen(x[rows], inv), rows)
        return self._frozen(x, inv)


class LRMFFWrapper(nn.Module):
    def __init__(
        self,
        transcoder: nn.Module = None,
        error_term: Tensor = None,
        timestep: Tensor = None,
        time_embed_dim: int = 256,
        linear_mode: bool = False,
        cached_output: Tensor = None,
        original_ff: nn.Module = None,
        branch: Optional[str] = None,
    ):
        super().__init__()

        self.transcoder = transcoder
        self.time_embed_dim = time_embed_dim
        self.linear_mode = linear_mode
        self.original_ff = original_ff
        self.branch = branch

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
            cached = self.cached_output.to(x.dtype)
            rows = traced_rows(x.shape[0], cached.shape[0], self.branch)
            if rows is not None:
                return _splice(self.original_ff(x), cached, rows)
            return cached

        rows = traced_rows(x.shape[0], self.error_term.shape[0], self.branch)
        if rows is not None:
            return _splice(self.original_ff(x), self._transcode(x[rows]), rows)
        return self._transcode(x)

    def _transcode(self, x: Tensor) -> Tensor:
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
        backend: Backend,
        original_forward: Optional[Callable] = None,
        branch: Optional[str] = None,
    ):
        super().__init__()
        self.original_attn = original_attn
        self.cache = cache
        self.backend = backend
        self.original_forward = original_forward
        self.branch = branch

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        rows = traced_rows(
            hidden_states.shape[0], self.cache.P.shape[0], self.branch
        )
        if rows is None:
            return self._frozen(hidden_states, encoder_hidden_states, **kwargs)

        out_img, out_txt = self.original_forward(
            hidden_states, encoder_hidden_states, **kwargs
        )
        frozen_img, frozen_txt = self._frozen(
            hidden_states[rows],
            encoder_hidden_states[rows],
            **{
                k: (v[rows] if torch.is_tensor(v) and v.shape[0] == hidden_states.shape[0] else v)
                for k, v in kwargs.items()
            },
        )
        return _splice(out_img, frozen_img, rows), _splice(out_txt, frozen_txt, rows)

    def _frozen(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        attn = self.original_attn
        device = hidden_states.device
        dtype = hidden_states.dtype
        B = hidden_states.shape[0]
        S_img, S_txt = self.cache.S_img, self.cache.S_txt
        num_heads, head_dim = self.cache.num_heads, self.cache.head_dim

        v_img = attn.to_v(hidden_states).view(B, S_img, num_heads, head_dim).transpose(1, 2)
        v_txt = (
            attn.add_v_proj(encoder_hidden_states)
            .view(B, S_txt, num_heads, head_dim)
            .transpose(1, 2)
        )

        V = self.backend.cat_joint(v_img, v_txt, dim=2)
        P = self.cache.get_P_on_device(device)

        out = torch.matmul(P, V.float()).to(dtype)
        out_img, out_txt = self.backend.split_joint(out, S_img, S_txt, dim=2)

        out_img = out_img.transpose(1, 2).reshape(B, S_img, -1)
        out_txt = out_txt.transpose(1, 2).reshape(B, S_txt, -1)

        out_img = self.backend.project_out_img(attn, out_img)
        out_txt = self.backend.project_out_txt(attn, out_txt)

        err_img, err_txt = self.cache.get_errors_on_device(device, dtype)
        return out_img + err_img, out_txt + err_txt


class FrozenSelfAttentionWrapper(nn.Module):
    def __init__(
        self,
        original_attn: nn.Module,
        cache: SelfAttentionCache,
        backend: Backend,
        original_forward: Optional[Callable] = None,
        branch: Optional[str] = None,
    ):
        super().__init__()
        self.original_attn = original_attn
        self.cache = cache
        self.backend = backend
        self.original_forward = original_forward
        self.branch = branch

    def forward(self, hidden_states: Tensor, **kwargs) -> Tensor:
        rows = traced_rows(
            hidden_states.shape[0], self.cache.P.shape[0], self.branch
        )
        if rows is not None:
            return _splice(
                self.original_forward(hidden_states, **kwargs),
                self._frozen(hidden_states[rows], **kwargs),
                rows,
            )
        return self._frozen(hidden_states, **kwargs)

    def _frozen(self, hidden_states: Tensor, **kwargs) -> Tensor:
        attn = self.original_attn
        device = hidden_states.device
        dtype = hidden_states.dtype
        B = hidden_states.shape[0]
        S_img = self.cache.S_img
        num_heads, head_dim = self.cache.num_heads, self.cache.head_dim

        V = attn.to_v(hidden_states).view(B, S_img, num_heads, head_dim).transpose(1, 2)
        P = self.cache.get_P_on_device(device)

        out = torch.matmul(P, V.float()).to(dtype)
        out = out.transpose(1, 2).reshape(B, S_img, -1)
        out = self.backend.project_out_img(attn, out)

        return out + self.cache.get_error_on_device(device, dtype)


class TraceCapturer:
    def __init__(
        self,
        pipe,
        transcoders: Dict[str, nn.Module],
        cfg: "LRMConfig",
    ):
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.transcoders = transcoders
        self.cfg = cfg
        self.backend = cfg.get_backend()

        self.hook_handles: List[Any] = []
        self.restore_fns: List[Callable] = []

    def _clear_hooks(self):
        for h in self.hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self.hook_handles.clear()

        for fn in self.restore_fns:
            try:
                fn()
            except Exception:
                pass
        self.restore_fns.clear()

    def _branch_bounds(self, batch_total: int) -> Optional[Tuple[int, int]]:
        if not self.backend.runs_real_cfg(self.cfg.guidance_scale):
            return None
        if batch_total < 2 or batch_total % 2 != 0:
            return None
        half = batch_total // 2
        if self.cfg.cfg_branch == "uncond":
            return (0, half)
        if self.cfg.cfg_branch == "cond":
            return (half, batch_total)
        raise ValueError(
            f"cfg_branch must be 'cond' or 'uncond', got {self.cfg.cfg_branch!r}"
        )

    @torch.no_grad()
    def capture(
        self,
        prompt: str,
        seed: int,
        target_step: int = 0,
    ) -> Trace:
        self._clear_hooks()
        backend = self.backend
        trace = Trace(
            prompt=prompt,
            seed=seed,
            step_idx=target_step,
            backend=self.cfg.backend,
        )

        step_state = {
            "transformer_call_idx": -1,
            "captured": False,
            "timestep": None,
            "branch": None,
            "batch_total": None,
        }

        def select(x):
            bounds = step_state["branch"]
            if bounds is None or not torch.is_tensor(x):
                return x
            if x.shape[0] != step_state["batch_total"]:
                return x
            return x[bounds[0] : bounds[1]]

        def transformer_pre_hook(module, args, kwargs):
            step_state["transformer_call_idx"] += 1
            current_step = step_state["transformer_call_idx"]

            if current_step != target_step or step_state["captured"]:
                return

            hs = kwargs.get("hidden_states", args[0] if args else None)
            batch_total = hs.shape[0] if hs is not None else 1
            step_state["batch_total"] = batch_total
            step_state["branch"] = self._branch_bounds(batch_total)
            trace.cfg_branch = (
                self.cfg.cfg_branch if step_state["branch"] is not None else None
            )

            trace.transformer_kwargs = {
                k: (select(v).detach().clone() if isinstance(v, Tensor) else v)
                for k, v in kwargs.items()
            }

            timestep = kwargs.get("timestep")
            if timestep is not None:
                timestep = select(timestep).detach().clone()
                scaled = timestep.to(dtype=torch.float32) / self.cfg.timestep_scale
                step_state["timestep"] = scaled
                trace.timestep_tensor = scaled.cpu()
                trace.timestep = float(scaled.mean().item())
                trace.timestep_raw = float(timestep.float().mean().item())

        self.hook_handles.append(
            self.transformer.register_forward_pre_hook(
                transformer_pre_hook, with_kwargs=True
            )
        )

        def active() -> bool:
            return (
                not step_state["captured"]
                and step_state["transformer_call_idx"] == target_step
            )

        def make_block_pre_hook(layer_idx: int):
            def hook(module, args, kwargs):
                if not active():
                    return

                hs = kwargs.get("hidden_states", args[0] if args else None)
                ehs = kwargs.get(
                    "encoder_hidden_states", args[1] if len(args) > 1 else None
                )

                if layer_idx == self.cfg.first_layer:
                    trace.boundary_img = select(hs).detach().cpu()
                    trace.boundary_txt = select(ehs).detach().cpu()
                    trace.S_img = hs.shape[1]
                    trace.S_txt = ehs.shape[1]

            return hook

        def make_adaln_hook(layer_idx: int, stream: str):
            def hook(module, args, kwargs, output):
                if not active():
                    return

                lc = trace.get_layer(layer_idx)
                prefix = "img" if stream == "img" else "txt"

                if isinstance(output, tuple) and len(output) >= 5:
                    _, gate_msa, shift_mlp, scale_mlp, gate_mlp = output[:5]
                    for name, val in (
                        ("gate_msa", gate_msa),
                        ("gate_mlp", gate_mlp),
                        ("scale_mlp", scale_mlp),
                        ("shift_mlp", shift_mlp),
                    ):
                        setattr(
                            lc,
                            f"{prefix}_{name}",
                            select(val).detach().cpu().to(torch.float32),
                        )
                    if len(output) >= 7:
                        lc.img_gate_msa2 = (
                            select(output[6]).detach().cpu().to(torch.float32)
                        )

                emb = kwargs.get("emb")
                if emb is None and len(args) > 1:
                    emb = args[1]
                if emb is not None and hasattr(module, "linear"):
                    with torch.no_grad():
                        mod_out = module.linear(module.silu(emb))
                    shift_msa2 = None
                    if isinstance(output, tuple) and len(output) >= 7:
                        chunks = mod_out.chunk(9, dim=-1)
                        shift_msa, shift_msa2 = chunks[0], chunks[6]
                    elif isinstance(output, tuple):
                        shift_msa = mod_out.chunk(6, dim=-1)[0]
                    else:
                        shift_msa = mod_out.chunk(2, dim=-1)[1]
                    setattr(
                        lc,
                        f"{prefix}_shift_msa",
                        select(shift_msa).detach().cpu().to(torch.float32),
                    )
                    if shift_msa2 is not None and stream == "img":
                        lc.img_shift_msa2 = (
                            select(shift_msa2).detach().cpu().to(torch.float32)
                        )

                x = args[0]
                inner_norm = getattr(module, "norm", None)
                eps = getattr(inner_norm, "eps", 1e-6)
                inv_denom = select(compute_layernorm_inv_denom(x, eps))

                if stream == "img":
                    lc.img_norm1_inv_denom = inv_denom.detach().cpu().to(torch.float32)
                else:
                    lc.txt_norm1_inv_denom = inv_denom.detach().cpu().to(torch.float32)

            return hook

        def make_norm2_hook(layer_idx: int, stream: str):
            def hook(module, args, output):
                if not active():
                    return

                x = args[0]
                eps = getattr(module, "eps", 1e-6)
                inv_denom = select(compute_layernorm_inv_denom(x, eps))

                lc = trace.get_layer(layer_idx)
                if stream == "img":
                    lc.img_norm2_inv_denom = inv_denom.detach().cpu().to(torch.float32)
                else:
                    lc.txt_norm2_inv_denom = inv_denom.detach().cpu().to(torch.float32)

            return hook

        def make_ff_hook(layer_idx: int, stream: str):
            def hook(module, args, output):
                if not active():
                    return

                x = select(args[0])
                y_true = select(output).float()

                tc = self.transcoders.get(f"{stream}_{layer_idx}")
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
                prefix = "img" if stream == "img" else "txt"
                setattr(lc, f"{prefix}_x_ff", x.detach().cpu().to(torch.float32))
                setattr(lc, f"{prefix}_z", z.detach().cpu().to(torch.float32))
                setattr(lc, f"{prefix}_h_pre", h_pre.detach().cpu().to(torch.float32))
                setattr(lc, f"{prefix}_ff_error", error.detach().cpu().to(torch.float32))
                setattr(lc, f"{prefix}_ff_out", y_true.detach().cpu())

            return hook

        def make_attn_wrapper(layer_idx: int, original_forward):
            def wrapped_forward(hidden_states, encoder_hidden_states=None, **kwargs):
                out_img, out_txt = original_forward(
                    hidden_states, encoder_hidden_states, **kwargs
                )
                if not active():
                    return out_img, out_txt

                blk = backend.blocks(self.transformer)[layer_idx]
                trace.get_layer(layer_idx).attention = capture_attention_with_errors(
                    backend=backend,
                    attn=blk.attn,
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    original_out_img=out_img,
                    original_out_txt=out_txt,
                    select=select,
                    **backend.attn_extra_kwargs(kwargs),
                )
                return out_img, out_txt

            return wrapped_forward

        def make_attn2_wrapper(layer_idx: int, original_forward):
            def wrapped_forward(hidden_states, **kwargs):
                out = original_forward(hidden_states, **kwargs)
                if not active():
                    return out

                blk = backend.blocks(self.transformer)[layer_idx]
                trace.get_layer(layer_idx).attention2 = (
                    capture_self_attention_with_errors(
                        backend=backend,
                        attn=blk.attn2,
                        hidden_states=hidden_states,
                        original_out=out,
                        select=select,
                    )
                )
                return out

            return wrapped_forward

        def step_callback(pipe, step_idx, timestep, callback_kwargs):
            if step_state["transformer_call_idx"] >= target_step:
                step_state["captured"] = True
            return callback_kwargs

        try:
            blocks = backend.blocks(self.transformer)
            for layer_idx in self.cfg.target_layers:
                blk = blocks[layer_idx]

                self.hook_handles.append(
                    blk.register_forward_pre_hook(
                        make_block_pre_hook(layer_idx), with_kwargs=True
                    )
                )

                for stream in ("img", "txt"):
                    norm1 = backend.norm1(blk, stream)
                    if norm1 is not None:
                        self.hook_handles.append(
                            norm1.register_forward_hook(
                                make_adaln_hook(layer_idx, stream), with_kwargs=True
                            )
                        )

                    norm2 = backend.norm2(blk, stream)
                    if norm2 is not None:
                        self.hook_handles.append(
                            norm2.register_forward_hook(
                                make_norm2_hook(layer_idx, stream)
                            )
                        )

                    ff = backend.ff(blk, stream)
                    if ff is not None:
                        self.hook_handles.append(
                            ff.register_forward_hook(make_ff_hook(layer_idx, stream))
                        )

                original_forward = blk.attn.forward
                blk.attn.forward = make_attn_wrapper(layer_idx, original_forward)
                self.restore_fns.append(
                    lambda blk=blk, orig=original_forward: setattr(
                        blk.attn, "forward", orig
                    )
                )

                attn2 = backend.dual_attn(blk)
                if attn2 is not None:
                    original_forward2 = attn2.forward
                    attn2.forward = make_attn2_wrapper(layer_idx, original_forward2)
                    self.restore_fns.append(
                        lambda a=attn2, orig=original_forward2: setattr(
                            a, "forward", orig
                        )
                    )

            generator = torch.Generator(device=self.cfg.device).manual_seed(seed)
            _ = self.pipe(
                prompt,
                height=self.cfg.height,
                width=self.cfg.width,
                num_inference_steps=self.cfg.num_inference_steps,
                generator=generator,
                callback_on_step_end=step_callback,
                output_type="latent",
                **self.cfg.generation_kwargs(prompt),
            )
        finally:
            self._clear_hooks()

        return trace


class LRMPatcher:
    def __init__(
        self,
        transformer: nn.Module,
        trace: Trace,
        transcoders: Dict[str, nn.Module],
        cfg: "LRMConfig",
        mode: str = "linear",
    ):
        self.transformer = transformer
        self.trace = trace
        self.transcoders = transcoders
        self.cfg = cfg
        self.mode = mode
        self.backend = cfg.get_backend()
        self.branch = trace.cfg_branch

        self._originals: Dict[int, Dict[str, Any]] = {}
        self._patched = False

    def _patch_ff(self, blk, lc, layer_idx: int, stream: str, linear_mode: bool):
        backend = self.backend
        original_ff = backend.ff(blk, stream)
        if original_ff is None:
            return

        device = self.cfg.device
        attr = "ff" if stream == "img" else "ff_context"
        cached = getattr(lc, f"{stream}_ff_out")

        if linear_mode and cached is not None:
            setattr(
                blk,
                attr,
                LRMFFWrapper(
                    cached_output=cached.to(device),
                    linear_mode=True,
                    original_ff=original_ff,
                    branch=self.branch,
                ),
            )
            return

        tc = self.transcoders.get(f"{stream}_{layer_idx}")
        error = getattr(lc, f"{stream}_ff_error")
        if tc is not None and error is not None:
            setattr(
                blk,
                attr,
                LRMFFWrapper(
                    transcoder=tc,
                    error_term=error.to(device),
                    timestep=self.trace.timestep_tensor.to(device),
                    time_embed_dim=self.cfg.time_embed_dim,
                    linear_mode=linear_mode,
                    original_ff=original_ff,
                    branch=self.branch,
                ),
            )

    def _patch_layer(self, layer_idx: int):
        backend = self.backend
        blk = backend.blocks(self.transformer)[layer_idx]
        lc = self.trace.get_layer(layer_idx)
        attn2 = backend.dual_attn(blk)

        self._originals[layer_idx] = {
            "ff": blk.ff,
            "ff_context": getattr(blk, "ff_context", None),
            "norm2": blk.norm2,
            "norm2_context": getattr(blk, "norm2_context", None),
            "norm1_norm": blk.norm1.norm,
            "norm1_context_norm": getattr(
                getattr(blk, "norm1_context", None), "norm", None
            ),
            "attn_forward": blk.attn.forward,
            "attn2_forward": attn2.forward if attn2 is not None else None,
        }

        device = self.cfg.device
        linear_mode = self.mode == "linear"

        for stream in ("img", "txt"):
            self._patch_ff(blk, lc, layer_idx, stream, linear_mode)
            norm2 = backend.norm2(blk, stream)
            inv2 = getattr(lc, f"{stream}_norm2_inv_denom")
            if norm2 is not None and inv2 is not None:
                attr = "norm2" if stream == "img" else "norm2_context"
                setattr(
                    blk,
                    attr,
                    FrozenNormWrapper(norm2, inv2.to(device), branch=self.branch),
                )

            norm1 = backend.norm1(blk, stream)
            inv1 = getattr(lc, f"{stream}_norm1_inv_denom")
            if norm1 is not None and getattr(norm1, "norm", None) is not None and inv1 is not None:
                norm1.norm = FrozenNormWrapper(
                    norm1.norm, inv1.to(device), branch=self.branch
                )

        if lc.attention is not None:
            wrapper = FrozenAttentionWrapper(
                blk.attn,
                lc.attention,
                backend,
                original_forward=self._originals[layer_idx]["attn_forward"],
                branch=self.branch,
            )

            def wrapped(hidden_states, encoder_hidden_states=None, _w=wrapper, **kwargs):
                return _w(hidden_states, encoder_hidden_states, **kwargs)

            blk.attn.forward = wrapped

        if attn2 is not None and lc.attention2 is not None:
            wrapper2 = FrozenSelfAttentionWrapper(
                attn2,
                lc.attention2,
                backend,
                original_forward=self._originals[layer_idx]["attn2_forward"],
                branch=self.branch,
            )

            def wrapped2(hidden_states, _w=wrapper2, **kwargs):
                return _w(hidden_states, **kwargs)

            attn2.forward = wrapped2

    def _restore_layer(self, layer_idx: int):
        if layer_idx not in self._originals:
            return

        orig = self._originals[layer_idx]
        blk = self.backend.blocks(self.transformer)[layer_idx]

        blk.ff = orig["ff"]
        blk.norm2 = orig["norm2"]
        blk.norm1.norm = orig["norm1_norm"]
        blk.attn.forward = orig["attn_forward"]

        if orig["ff_context"] is not None:
            blk.ff_context = orig["ff_context"]
        if orig["norm2_context"] is not None:
            blk.norm2_context = orig["norm2_context"]
        if orig["norm1_context_norm"] is not None:
            blk.norm1_context.norm = orig["norm1_context_norm"]
        if orig["attn2_forward"] is not None:
            blk.attn2.forward = orig["attn2_forward"]

    def __enter__(self):
        if self._patched:
            raise RuntimeError("This LRMPatcher is already active.")
        try:
            for layer_idx in self.cfg.target_layers:
                self._patch_layer(layer_idx)
        except BaseException:
            for layer_idx in self.cfg.target_layers:
                self._restore_layer(layer_idx)
            self._originals.clear()
            raise
        self._patched = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for layer_idx in self.cfg.target_layers:
            self._restore_layer(layer_idx)
        self._originals.clear()
        self._patched = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False


def load_pipeline(cfg: LRMConfig):
    PipelineCls = getattr(diffusers, cfg.pipeline_cls)
    return PipelineCls.from_pretrained(cfg.model_id, torch_dtype=cfg.dtype).to(cfg.device)
