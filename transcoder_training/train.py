import os
import random
import diffusers
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from .arch import ARCHS, runs_real_cfg
from .transcoder import TemporalAwareTranscoder, TemporalAwareSAE
from .activation_store import (
    TimestepContext,
    DualStreamCapture,
    install_timestep_hook,
    make_buffers,
    reset_buffers,
)
from .data import PromptStream
from .evaluation import run_validation


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig:
    pipeline_cls: str = "FluxPipeline"
    model_id: str = "black-forest-labs/FLUX.1-schnell"

    dataset_id: str = "yvdao/midjourney-v6"
    dataset_column: str = "prompt"
    dataset_config: Optional[str] = "default"
    dataset_split: str = "train"

    target_layers: Tuple[int, ...] = (6, 12, 18)
    d_model: int = 3072
    expansion_factor: int = 16
    l1_coeff: Dict[str, float] = field(
        default_factory=lambda: {"img": 3e-4, "txt": 5e-5}
    )
    lr: Dict[str, float] = field(
        default_factory=lambda: {"img": 2e-4, "txt": 2e-4}
    )
    time_embed_dim: int = 256

    num_inference_steps: int = 4
    guidance_scale: float = 0.0
    timestep_scale: float = 1.0
    height: int = 512
    width: int = 512
    prompt_aliases: Tuple[str, ...] = ("prompt_2",)

    buffer_size: int = 1_000_000
    batch_size: int = 4096
    total_cycles: int = 256
    prompts_per_inference: int = 32

    val_prompts: int = 512
    val_every: int = 16
    stats_every: int = 8
    make_comparison_image: bool = True

    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    train_dtype: torch.dtype = torch.float32
    buffer_dtype: torch.dtype = torch.bfloat16

    save_dir: str = "./output"
    seed: int = 42

    @classmethod
    def for_model(cls, name: str, **overrides) -> "TrainConfig":
        spec = ARCHS.get(name, None)
        if not spec:
            raise ValueError(f"Unknown model: {name}")

        base: Dict[str, Any] = dict(
            model_id=spec.default_model_id,
            pipeline_cls=spec.pipeline_cls,
            d_model=spec.d_model,
            guidance_scale=spec.guidance_scale,
            num_inference_steps=spec.num_inference_steps,
            timestep_scale=spec.timestep_scale,
            prompt_aliases=spec.prompt_aliases,
            buffer_size=spec.buffer_size,
            prompts_per_inference=spec.prompts_per_inference,
        )
        if spec.l1_coeff is not None:
            base["l1_coeff"] = dict(spec.l1_coeff)
        if spec.lr is not None:
            base["lr"] = dict(spec.lr)

        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**base)


def _build_models(
    cfg: TrainConfig, keys: Sequence[Tuple[str, int]], role: str
) -> nn.ModuleDict:
    cls = TemporalAwareSAE if role == "sae" else TemporalAwareTranscoder
    models = nn.ModuleDict()
    for stream, l in keys:
        models[f"{stream}_{l}"] = cls(
            cfg.d_model,
            cfg.expansion_factor,
            cfg.time_embed_dim,
            l1_coeff=cfg.l1_coeff[stream],
        )
    return models.to(cfg.device).to(cfg.train_dtype)


def _validate_target_layers(pipe, cfg: TrainConfig, keys: Sequence[Tuple[str, int]]) -> None:
    blocks = pipe.transformer.transformer_blocks
    for stream, layer in keys:
        if layer >= len(blocks):
            raise ValueError(
                f"target layer {layer} is out of range: "
                f"{cfg.model_id} has {len(blocks)} layers."
            )

        module = blocks[layer].ff if stream == "img" else blocks[layer].ff_context
        if module is None:
            raise ValueError(
                f"block {layer} of {cfg.model_id} has no '{stream}' MLP."
            )

    d_actual = getattr(pipe.transformer, "inner_dim", None) or (
        pipe.transformer.config.num_attention_heads * pipe.transformer.config.attention_head_dim
    )
    if d_actual != cfg.d_model:
        raise ValueError(
            f"d_model mismatch: config says {cfg.d_model} but "
            f"{cfg.model_id} has a residual stream of {d_actual}."
        )


def _check_trajectory_coverage(cfg):
    assert cfg.height % 16 == 0 and cfg.width % 16 == 0
    s_img = (cfg.height // 16) * (cfg.width // 16)
    mul = 2 if runs_real_cfg(cfg.pipeline_cls, cfg.guidance_scale) else 1
    harvest_batch = cfg.prompts_per_inference * mul

    rows_per_step = harvest_batch * s_img
    rows_per_call = rows_per_step * cfg.num_inference_steps
    max_prompts = cfg.buffer_size // (s_img * cfg.num_inference_steps * mul)

    if rows_per_call > cfg.buffer_size:
        covered = min(cfg.num_inference_steps, -(-cfg.buffer_size // rows_per_step))
        raise ValueError(
            f"harvest would truncate the noise schedule so only steps "
            f"0...{covered - 1} of {cfg.num_inference_steps} are ever stored.\n"
            f"fix by lowering prompts_per_inference to <= {max_prompts}, "
            f"or raising buffer_size to >= {rows_per_call:,}"
        )


def run_training(cfg: TrainConfig, role: str = "transcoder"):
    """Train all (layer, stream) dictionaries, role in {"transcoder", "sae"}."""
    assert role in ("transcoder", "sae")
    cfg = replace(cfg)
    _check_trajectory_coverage(cfg)

    seed_everything(cfg.seed)
    os.makedirs(os.path.join(cfg.save_dir, "best"), exist_ok=True)
    os.makedirs(os.path.join(cfg.save_dir, "last"), exist_ok=True)

    PipelineCls = getattr(diffusers, cfg.pipeline_cls)
    pipe = PipelineCls.from_pretrained(
        cfg.model_id, torch_dtype=cfg.dtype
    ).to(cfg.device)
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.requires_grad_(False)
    pipe.transformer.eval()

    keys = [(s, l) for l in cfg.target_layers for s in ("img", "txt")]
    _validate_target_layers(pipe, cfg, keys)
    models = _build_models(cfg, keys, role)

    device_type = torch.device(cfg.device).type
    autocast_enabled = device_type == "cuda"
    t_ctx = TimestepContext(scale=cfg.timestep_scale)

    hook_handle = install_timestep_hook(pipe, t_ctx)
    buffers = make_buffers(keys, cfg.d_model, cfg.buffer_size, cfg.buffer_dtype)
    capturer = DualStreamCapture(pipe, keys, buffers, cfg.buffer_dtype, t_ctx)

    stream = PromptStream(
        cfg.dataset_id,
        column=cfg.dataset_column,
        split=cfg.dataset_split,
        config_name=cfg.dataset_config,
    )
    val_prompts = stream.fixed_validation_batch(n=cfg.val_prompts)

    optimizers, schedulers = {}, {}
    for key, model in models.items():
        s = key.split("_")[0]
        opt = optim.AdamW(model.parameters(), lr=cfg.lr[s], weight_decay=0)
        optimizers[key] = opt
        schedulers[key] = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.total_cycles
        )

    first_key = f"img_{cfg.target_layers[0]}"
    nsteps = cfg.buffer_size // cfg.batch_size
    best_val_cos = 0.0

    for cycle in tqdm(range(cfg.total_cycles), desc=f"Training {role}"):
        reset_buffers(buffers)

        # harvest activations until the buffer is full
        capturer.enabled = True
        pbar = tqdm(total=cfg.buffer_size, desc="Collecting", leave=False)

        def _harvest_done() -> bool:
            return buffers[first_key]["ptr"] >= cfg.buffer_size

        while not _harvest_done():
            prompts = stream.get_prompts(cfg.prompts_per_inference)
            before = int(buffers[first_key]["ptr"])
            call_kwargs = {alias: prompts for alias in cfg.prompt_aliases}
            with torch.inference_mode():
                pipe(
                    prompts,
                    height=cfg.height,
                    width=cfg.width,
                    num_inference_steps=cfg.num_inference_steps,
                    guidance_scale=cfg.guidance_scale,
                    output_type="latent",
                    **call_kwargs,
                )
            pbar.update(max(0, int(buffers[first_key]["ptr"]) - before))

        pbar.close()
        if device_type == "cuda":
            torch.cuda.synchronize()

        # one optimizer epoch per (layer, stream)
        models.train()
        stats = {k: {"nmse": 0.0, "l0": 0.0} for k in models.keys()}
        for key in tqdm(list(models.keys()), desc="Backprop", leave=False):
            buf, model = buffers[key], models[key]
            s = key.split("_")[0]
            opt = optimizers[key]
            for _ in tqdm(range(nsteps), desc=key, leave=False):
                idx = torch.randint(0, buf["ptr"], (cfg.batch_size,), device="cpu")
                bx = buf["x"][idx].to(cfg.device, non_blocking=True).float()
                by = buf["y"][idx].to(cfg.device, non_blocking=True).float()
                bt = buf["t"][idx].to(cfg.device, non_blocking=True)

                model_in = by if role == "sae" else bx
                target = by

                opt.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device_type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    rec, z = model(model_in, bt)

                diff = target - rec.float()
                mse = diff.pow(2).sum(dim=-1).mean()
                target_var = target.var(dim=0, unbiased=False).sum() + 1e-6
                normalized_mse = mse / target_var
                sparsity = z.float().abs().sum(dim=-1).mean()
                loss = normalized_mse + cfg.l1_coeff[s] * sparsity

                loss.backward()
                opt.step()
                model.normalize_decoder()

                stats[key]["nmse"] += normalized_mse.item()
                stats[key]["l0"] += (z > 0).float().sum(dim=-1).mean().item()

        for sched in schedulers.values():
            sched.step()

        if (cycle + 1) % cfg.stats_every == 0:
            tqdm.write(f"cycle {cycle + 1}")
            for k in models.keys():
                s, l = k.split("_")
                tqdm.write(
                    f"L{l} {s.upper()} | nMSE: {stats[k]['nmse']/nsteps:.4f} | L0: {stats[k]['l0']/nsteps:.1f}"
                )

        if (cycle + 1) % cfg.val_every == 0:
            val_mse, val_cos, val_image = run_validation(
                pipe,
                models,
                keys,
                val_prompts,
                capturer,
                t_ctx,
                kind=role,
                num_inference_steps=cfg.num_inference_steps,
                device=cfg.device,
                height=cfg.height,
                width=cfg.width,
                orig_dtype=cfg.dtype,
                guidance_scale=cfg.guidance_scale,
                prompt_aliases=cfg.prompt_aliases,
                make_comparison_image=cfg.make_comparison_image,
            )
            tqdm.write(f"[val] cos={val_cos:.4f} mse={val_mse:.4f}")
            if val_image is not None:
                val_image.save(os.path.join(cfg.save_dir, f"val_cycle_{cycle + 1}.png"))
            if val_cos > best_val_cos:
                best_val_cos = val_cos
                for key, model in models.items():
                    torch.save(
                        model.state_dict(),
                        os.path.join(cfg.save_dir, "best", f"{role}_{key}.pt"),
                    )

    hook_handle.remove()
    capturer.close()

    for key, model in models.items():
        torch.save(
            model.state_dict(), os.path.join(cfg.save_dir, "last", f"{role}_{key}.pt")
        )
    return models
