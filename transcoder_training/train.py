import os
import random
from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from .transcoder import TemporalAwareTranscoder, TemporalAwareSAE
from .activation_store import (
    TimestepContext,
    DualStreamCapture,
    install_timestep_hook,
    make_buffers,
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
    model_id: str = "black-forest-labs/FLUX.1-schnell"
    dataset_id: str = "yvdao/midjourney-v6"
    dataset_column: str = "prompt"
    target_layers: Tuple[int, ...] = (6, 12, 18)
    d_model: int = 3072
    expansion_factor: int = 16
    l1_coeff: Dict[str, float] = field(
        default_factory=lambda: {"img": 3e-4, "txt": 5e-5}
    )
    lr: Dict[str, float] = field(default_factory=lambda: {"img": 2e-4, "txt": 2e-4})
    time_embed_dim: int = 256
    num_inference_steps: int = 4
    height: int = 512
    width: int = 512
    buffer_size: int = 1_000_000
    batch_size: int = 4096
    total_cycles: int = 256
    prompts_per_inference: int = 32
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    train_dtype: torch.dtype = torch.float32
    buffer_dtype: torch.dtype = torch.bfloat16
    save_dir: str = "./output"
    seed: int = 42


def _build_models(cfg: TrainConfig, role: str) -> nn.ModuleDict:
    Cls = TemporalAwareSAE if role == "sae" else TemporalAwareTranscoder
    models = nn.ModuleDict()
    for l in cfg.target_layers:
        for stream in ("img", "txt"):
            models[f"{stream}_{l}"] = Cls(
                cfg.d_model,
                cfg.expansion_factor,
                cfg.time_embed_dim,
                l1_coeff=cfg.l1_coeff[stream],
            )
    return models.to(cfg.device).to(cfg.train_dtype)


def run_training(cfg: TrainConfig, role: str = "transcoder"):
    """Train all (layer, stream) dictionaries, role in {"transcoder", "sae"}."""
    assert role in ("transcoder", "sae")
    from diffusers import FluxPipeline

    seed_everything(cfg.seed)
    os.makedirs(os.path.join(cfg.save_dir, "best"), exist_ok=True)
    os.makedirs(os.path.join(cfg.save_dir, "last"), exist_ok=True)

    pipe = FluxPipeline.from_pretrained(cfg.model_id, torch_dtype=cfg.dtype).to(
        cfg.device
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.requires_grad_(False)
    pipe.transformer.eval()

    models = _build_models(cfg, role)

    t_ctx = TimestepContext()
    install_timestep_hook(pipe, t_ctx)
    buffers = make_buffers(
        cfg.target_layers, cfg.d_model, cfg.buffer_size, cfg.buffer_dtype
    )
    capturer = DualStreamCapture(
        pipe, cfg.target_layers, buffers, cfg.buffer_dtype, t_ctx
    )

    stream = PromptStream(cfg.dataset_id, column=cfg.dataset_column)
    val_prompts = stream.fixed_validation_batch(n=512)

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
        for key in buffers:
            buffers[key]["ptr"] = 0

        # harvest activations until the buffer is full
        capturer.enabled = True
        pbar = tqdm(total=cfg.buffer_size, desc="Collecting", leave=False)
        while buffers[first_key]["ptr"] < cfg.buffer_size:
            prompts = stream.get_prompts(cfg.prompts_per_inference)
            before = int(buffers[first_key]["ptr"])
            with torch.inference_mode():
                pipe(
                    prompts,
                    prompt_2=prompts,
                    height=cfg.height,
                    width=cfg.width,
                    num_inference_steps=cfg.num_inference_steps,
                    guidance_scale=0.0,
                    output_type="latent",
                )
            pbar.update(max(0, int(buffers[first_key]["ptr"]) - before))
        pbar.close()
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
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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

        if (cycle + 1) % 8 == 0:
            tqdm.write(f"cycle {cycle + 1}")
            for l in cfg.target_layers:
                for s in ("img", "txt"):
                    k = f"{s}_{l}"
                    tqdm.write(
                        f"L{l} {s.upper()} | nMSE: {stats[k]['nmse']/nsteps:.4f} | L0: {stats[k]['l0']/nsteps:.1f}"
                    )

        if (cycle + 1) % 16 == 0:
            val_mse, val_cos, _ = run_validation(
                pipe,
                models,
                cfg.target_layers,
                val_prompts,
                capturer,
                t_ctx,
                kind=role,
                num_inference_steps=cfg.num_inference_steps,
                device=cfg.device,
                height=cfg.height,
                width=cfg.width,
            )
            tqdm.write(f"[val] cos={val_cos:.4f} mse={val_mse:.4f}")
            if val_cos > best_val_cos:
                best_val_cos = val_cos
                for key, model in models.items():
                    torch.save(
                        model.state_dict(),
                        os.path.join(cfg.save_dir, "best", f"{role}_{key}.pt"),
                    )

    for key, model in models.items():
        torch.save(
            model.state_dict(), os.path.join(cfg.save_dir, "last", f"{role}_{key}.pt")
        )
    return models
