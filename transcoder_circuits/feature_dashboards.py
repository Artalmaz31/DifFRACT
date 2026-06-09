import os
import io
import gc
import base64
import logging
import html as html_lib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
import transformers
from datasets import load_dataset
from diffusers import FluxPipeline
from transcoder_training.transcoder import load_transcoders
from .pipeline import FluxLRMPipeline
from .replacement_model import FluxTrace
from IPython.display import display, HTML

CONFIG = {
    "model_id": "black-forest-labs/FLUX.1-schnell",
    "dataset_id": "yvdao/midjourney-v6",
    "dataset_column": "prompt",
    "target_layers": [6, 12, 18],
    "dims": {"img": 3072, "txt": 3072},
    "expansion_factor": 16,
    "time_embed_dim": 256,
    "device": "cuda",
    "dtype": torch.bfloat16,
    "transcoder_dir": os.environ.get("TRANSCODERS_DIR", "transcoders"),
    "num_prompts_scan": 100_000,
    "height": 512,
    "width": 512,
    "batch_size": 64,
    "seed_base": 42,
    "num_inference_steps": 4,
    "top_k_per_feature": 5,
    "winner_top_m": 128,
    "batch_features": 2048,
    "max_features_per_key": 256,
    "save_images": True,
    "num_features_to_show": 128,
    "min_frac_of_max": 0.20,
    "max_alpha": 0.90,
    "ignore_special_tokens": False,
    "special_tokens": {"</s>"},
}


@dataclass
class TopKMeta:
    prompt_idx: torch.Tensor
    seed: torch.Tensor
    step_idx: torch.Tensor


class TopKTable:
    def __init__(self, n_features, k, device):
        self.F = n_features
        self.K = k
        self.device = device

        self.values = torch.full(
            (self.F, self.K), -float("inf"), device=device, dtype=torch.float32
        )
        self.meta = TopKMeta(
            prompt_idx=torch.full(
                (self.F, self.K), -1, device=device, dtype=torch.int32
            ),
            seed=torch.full((self.F, self.K), -1, device=device, dtype=torch.int32),
            step_idx=torch.full((self.F, self.K), -1, device=device, dtype=torch.int16),
        )

        self.total_seen = 0
        self.sum_vals = torch.zeros((self.F,), device=device, dtype=torch.float32)
        self.sumsq_vals = torch.zeros((self.F,), device=device, dtype=torch.float32)
        self.winner_count = torch.zeros((self.F,), device=device, dtype=torch.int32)

    @torch.no_grad()
    def update_topk(self, batch_vals, prompt_idx, seed, step_idx):
        B, F = batch_vals.shape
        assert F == self.F

        self.total_seen += B
        self.sum_vals += batch_vals.sum(dim=0).float()
        self.sumsq_vals += (batch_vals.float() ** 2).sum(dim=0)

        old_vals = self.values
        new_vals = batch_vals.transpose(0, 1).contiguous()
        merged_vals = torch.cat([old_vals, new_vals], dim=1)

        top_vals, top_idx = torch.topk(
            merged_vals, k=self.K, dim=1, largest=True, sorted=True
        )

        new_prompt = prompt_idx.view(1, B).expand(F, B).to(torch.int32)
        new_seed = seed.view(1, B).expand(F, B).to(torch.int32)
        new_step = torch.full(
            (F, B), int(step_idx), device=self.device, dtype=torch.int16
        )

        merged_prompt = torch.cat([self.meta.prompt_idx, new_prompt], dim=1)
        merged_seed = torch.cat([self.meta.seed, new_seed], dim=1)
        merged_step = torch.cat([self.meta.step_idx, new_step], dim=1)

        self.values = top_vals
        self.meta = TopKMeta(
            prompt_idx=torch.gather(merged_prompt, 1, top_idx),
            seed=torch.gather(merged_seed, 1, top_idx),
            step_idx=torch.gather(merged_step, 1, top_idx),
        )

    @torch.no_grad()
    def update_winners(self, batch_vals, top_m):
        top_m = min(top_m, batch_vals.shape[1])
        _, idx = torch.topk(batch_vals, k=top_m, dim=1, largest=True, sorted=False)
        flat = idx.reshape(-1)
        ones = torch.ones_like(flat, dtype=torch.int32)
        self.winner_count.scatter_add_(0, flat, ones)

    def to_cpu(self):
        return {
            "F": self.F,
            "K": self.K,
            "total_seen": int(self.total_seen),
            "values": self.values.detach().cpu().numpy(),
            "prompt_idx": self.meta.prompt_idx.detach().cpu().numpy(),
            "seed": self.meta.seed.detach().cpu().numpy(),
            "step_idx": self.meta.step_idx.detach().cpu().numpy(),
            "sum_vals": self.sum_vals.detach().cpu().numpy(),
            "sumsq_vals": self.sumsq_vals.detach().cpu().numpy(),
            "winner_count": self.winner_count.detach().cpu().numpy(),
        }


class ActivationCapturer:
    def __init__(
        self,
        pipe,
        layers,
        transcoders,
        n_inference_steps,
        capture_steps,
        max_only,
        batch_features=2048,
        subset_feature_idx=None,
    ):
        self.pipe = pipe
        self.layers = layers
        self.transcoders = transcoders
        self.n_steps = n_inference_steps
        self.capture_steps = capture_steps
        self.capture_set = set(capture_steps)
        self.max_only = max_only
        self.batch_features = batch_features
        self.subset_feature_idx = subset_feature_idx or {}

        self.step_call_idx = -1
        self.current_timestep = None
        self.captured = {}
        self.hooks = []

        self.hooks.append(
            self.pipe.transformer.register_forward_pre_hook(
                self._on_transformer_pre, with_kwargs=True
            )
        )

        for l in layers:
            blk = pipe.transformer.transformer_blocks[l]
            self.hooks.append(blk.ff.register_forward_hook(self._make_hook(f"img_{l}")))
            self.hooks.append(
                blk.ff_context.register_forward_hook(self._make_hook(f"txt_{l}"))
            )

    def _on_transformer_pre(self, module, args, kwargs):
        self.step_call_idx += 1
        t = kwargs.get("timestep", None)
        self.current_timestep = t
        return None

    def _make_hook(self, key):
        def hook_fn(module, args, output):
            step = self.step_call_idx
            if step not in self.capture_set:
                return

            x = args[0]
            tc = self.transcoders[key]
            t = self.current_timestep

            with torch.no_grad():
                if key in self.subset_feature_idx:
                    feats = tc.encode_batch(x, t, self.subset_feature_idx[key])
                    if self.max_only:
                        feats = feats.amax(dim=1)
                else:
                    feats = tc.encode_max(x, t, batch=self.batch_features)

            self.captured.setdefault(key, {})[step] = feats.detach()

        return hook_fn

    def reset(self):
        self.step_call_idx = -1
        self.current_timestep = None
        self.captured = {}

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def load_prompts_stream(cfg, min_len=16, max_len=512):
    ds = load_dataset(cfg["dataset_id"], "default", split="train", streaming=True)
    prompts = []
    it = iter(ds)
    with tqdm(total=CONFIG["num_prompts_scan"], desc="Loading prompts") as pbar:
        while len(prompts) < cfg["num_prompts_scan"]:
            try:
                item = next(it)
            except StopIteration:
                it = iter(ds)
                continue
            txt = item.get(cfg["dataset_column"], "")
            if isinstance(txt, str) and len(txt) >= min_len:
                prompts.append(txt[:max_len])
                pbar.update(1)
    return prompts


@torch.inference_mode()
def pass_a_scan(pipe, transcoders, prompts, cfg):
    device = cfg["device"]
    bs = cfg["batch_size"]
    n_steps = cfg["num_inference_steps"]
    H, W = cfg["height"], cfg["width"]
    seed_base = cfg["seed_base"]
    capture_steps = list(range(n_steps))

    any_key = f"img_{cfg['target_layers'][0]}"
    n_features = transcoders[any_key].d_feat

    capturer = ActivationCapturer(
        pipe=pipe,
        layers=CONFIG["target_layers"],
        transcoders=transcoders,
        n_inference_steps=n_steps,
        capture_steps=capture_steps,
        max_only=True,
        batch_features=CONFIG["batch_features"],
        subset_feature_idx=None,
    )

    tables = {}
    for l in cfg["target_layers"]:
        for stream in ["img", "txt"]:
            key = f"{stream}_{l}"
            tables[key] = {
                st: TopKTable(n_features, cfg["top_k_per_feature"], device)
                for st in capture_steps
            }

    n_batches = (len(prompts) + bs - 1) // bs
    for b in tqdm(range(n_batches), desc="Pass A"):
        s = b * bs
        e = min(s + bs, len(prompts))
        batch_prompts = prompts[s:e]
        B = len(batch_prompts)

        seeds = torch.tensor(
            [seed_base + s + i for i in range(B)], device=device, dtype=torch.int32
        )
        prompt_idx = torch.tensor(list(range(s, e)), device=device, dtype=torch.int32)
        generators = [
            torch.Generator(device=device).manual_seed(int(x)) for x in seeds.tolist()
        ]

        capturer.reset()
        _ = pipe(
            batch_prompts,
            prompt_2=batch_prompts,
            height=H,
            width=W,
            num_inference_steps=n_steps,
            guidance_scale=0.0,
            output_type="latent",
            generator=generators,
        )

        for key, by_step in capturer.captured.items():
            for st, max_feats in by_step.items():
                tab = tables[key][st]
                tab.update_topk(
                    max_feats, prompt_idx=prompt_idx, seed=seeds, step_idx=st
                )
                tab.update_winners(max_feats, top_m=CONFIG["winner_top_m"])

        if b % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    capturer.close()

    out = {
        key: {st: tab.to_cpu() for st, tab in sd.items()} for key, sd in tables.items()
    }
    return out


def _safe_mean_std(tab):
    total = max(int(tab["total_seen"]), 1)
    mean = tab["sum_vals"] / total
    var = tab["sumsq_vals"] / total - mean**2
    std = np.sqrt(np.maximum(var, 1e-8))
    return mean, std


def select_features(scan, key, max_features_per_key=256):
    steps = sorted(scan[key].keys())
    if not steps:
        return [], {}

    F = scan[key][steps[0]]["F"]
    n_steps = len(steps)

    z = np.zeros((n_steps, F), dtype=np.float32)
    freq = np.zeros((n_steps, F), dtype=np.float32)
    mean_act = np.zeros((n_steps, F), dtype=np.float32)

    for si, st in enumerate(steps):
        tab = scan[key][st]
        total = max(int(tab["total_seen"]), 1)
        mean, std = _safe_mean_std(tab)
        top1 = tab["values"][:, 0].astype(np.float32)

        z_step = (top1 - mean) / (std + 1e-6)
        win_count = tab["winner_count"].astype(np.float32)
        freq_step = win_count / float(total)

        z[si] = z_step
        freq[si] = freq_step
        mean_act[si] = tab["sum_vals"] / total

    # Feature selection score
    score = z * np.sqrt(np.clip(freq, 0.0, None))
    score[~np.isfinite(score)] = -np.inf

    best_si = np.argmax(mean_act, axis=0)
    best_score = np.max(score, axis=0)

    finite = np.isfinite(best_score)
    idx_all = np.nonzero(finite)[0]
    if idx_all.size == 0:
        return [], {}

    K = min(max_features_per_key, idx_all.size)
    top_idx = np.argpartition(-best_score[idx_all], K - 1)[:K]
    top_feat = idx_all[top_idx]
    top_feat = top_feat[np.argsort(-best_score[top_feat])]

    features = [int(f) for f in top_feat]
    best_step = {int(f): int(steps[int(best_si[f])]) for f in features}
    return features, best_step


@dataclass
class PassBState:
    capture_steps: List[int]
    selected_features: Dict[str, List[int]]
    best_step: Dict[str, Dict[int, int]]
    features_per_key: Dict[str, List[int]]
    feat2col: Dict[str, Dict[int, int]]
    examples: List[Dict[str, int]]
    example_index: Dict[Tuple[int, int], int]
    acts: Dict[str, Dict[int, List[np.ndarray]]]
    images: Optional[List[Image.Image]]
    height: int
    width: int
    num_inference_steps: int


@dataclass(frozen=True)
class ExampleId:
    prompt_idx: int
    seed: int


@dataclass(frozen=True)
class ExampleActivation:
    prompt_idx: int
    seed: int
    step: int
    value: float


def top_examples_for_feature(scan, key, feature_idx, top_n=5):
    refs = []
    for st, tab in scan[key].items():
        vals = tab["values"][feature_idx]
        pidx = tab["prompt_idx"][feature_idx]
        seed = tab["seed"][feature_idx]
        for v, pi, sd in zip(vals, pidx, seed):
            pi = int(pi)
            sd = int(sd)
            if pi < 0 or sd < 0:
                continue
            refs.append(ExampleActivation(pi, sd, int(st), float(v)))

    best = {}
    for r in refs:
        k = (r.prompt_idx, r.seed)
        if k not in best or r.value > best[k].value:
            best[k] = r

    out = list(best.values())
    out.sort(key=lambda r: -r.value)
    return out[:top_n]


@torch.inference_mode()
def pass_b_extract(pipe, transcoders, prompts, scan, selected_features, best_step, cfg):
    device = cfg["device"]
    n_steps = cfg["num_inference_steps"]
    H, W = cfg["height"], cfg["width"]
    bs = cfg["batch_size"]

    capture_steps = list(range(n_steps))
    features_per_key = {}
    feat2col = {}

    for key, features in selected_features.items():
        feats = sorted({f for f in features})
        features_per_key[key] = feats
        feat2col[key] = {f: i for i, f in enumerate(feats)}

    subset_idx = {
        key: torch.tensor(feats, device=device, dtype=torch.long)
        for key, feats in features_per_key.items()
    }

    capturer = ActivationCapturer(
        pipe=pipe,
        layers=cfg["target_layers"],
        transcoders=transcoders,
        n_inference_steps=n_steps,
        capture_steps=capture_steps,
        max_only=False,
        batch_features=cfg["batch_features"],
        subset_feature_idx=subset_idx,
    )

    all_ex = set()
    for key, feats in selected_features.items():
        for feat_idx in feats:
            refs = top_examples_for_feature(
                scan, key, feat_idx, top_n=cfg["top_k_per_feature"]
            )
            for r in refs:
                all_ex.add(ExampleId(r.prompt_idx, r.seed))

    all_ex = sorted(list(all_ex), key=lambda x: (x.prompt_idx, x.seed))
    acts = {}
    for key in selected_features.keys():
        acts[key] = {st: [] for st in capture_steps}

    images = [] if cfg.get("save_images", True) else None
    examples = []
    n_batches = (len(all_ex) + bs - 1) // bs
    for b in tqdm(range(n_batches), desc="Pass B"):
        s = b * bs
        e = min(s + bs, len(all_ex))
        batch_ex = all_ex[s:e]

        batch_prompts = [prompts[x.prompt_idx] for x in batch_ex]
        generators = [
            torch.Generator(device=device).manual_seed(int(x.seed)) for x in batch_ex
        ]

        capturer.reset()
        out = pipe(
            batch_prompts,
            prompt_2=batch_prompts,
            height=H,
            width=W,
            num_inference_steps=n_steps,
            guidance_scale=0.0,
            output_type="pil" if cfg.get("save_images", True) else "latent",
            generator=generators,
        )

        for ex in batch_ex:
            examples.append({"prompt_idx": ex.prompt_idx, "seed": ex.seed})

        if images is not None:
            images.extend(list(out.images))

        for key, by_step in capturer.captured.items():
            for st, tens in by_step.items():
                arr = tens.detach().to(torch.float16).cpu().numpy()
                for i in range(arr.shape[0]):
                    acts[key][st].append(arr[i])

        if b % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    capturer.close()

    example_index = {}
    for i, ex in enumerate(examples):
        example_index[(ex["prompt_idx"], ex["seed"])] = i

    state = PassBState(
        capture_steps=capture_steps,
        selected_features=selected_features,
        best_step=best_step,
        features_per_key=features_per_key,
        feat2col=feat2col,
        examples=examples,
        example_index=example_index,
        acts=acts,
        images=images,
        height=H,
        width=W,
        num_inference_steps=n_steps,
    )

    return state


def run_pipeline(cfg):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    transformers.utils.logging.set_verbosity_error()
    logging.getLogger("diffusers").setLevel(logging.ERROR)

    pipe = FluxPipeline.from_pretrained(cfg["model_id"], torch_dtype=cfg["dtype"]).to(
        cfg["device"]
    )
    pipe.transformer.requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)

    transcoders = load_transcoders(
        cfg["transcoder_dir"],
        cfg["target_layers"],
        d_model=cfg["dims"]["img"],
        expansion_factor=cfg["expansion_factor"],
        time_embed_dim=cfg["time_embed_dim"],
        device=cfg["device"],
        dtype=cfg["dtype"],
    )

    prompts = load_prompts_stream(cfg)
    scan = pass_a_scan(pipe, transcoders, prompts, cfg)

    selected_features = {}
    best_step = {}
    for l in cfg["target_layers"]:
        for stream in ["img", "txt"]:
            key = f"{stream}_{l}"
            feats, bs = select_features(
                scan, key, max_features_per_key=cfg["max_features_per_key"]
            )
            selected_features[key] = feats
            best_step[key] = bs
            print(f"{key}: {len(feats)} features selected")

    pass_b_state = pass_b_extract(
        pipe, transcoders, prompts, scan, selected_features, best_step, cfg
    )
    return pipe, prompts, scan, selected_features, pass_b_state


def tokenize_for_display(pipe, prompt):
    enc = pipe.tokenizer_2(prompt, truncation=True, max_length=512, return_tensors="pt")
    ids = enc.input_ids[0].tolist()
    toks = pipe.tokenizer_2.convert_ids_to_tokens(ids)

    out = []
    for t in toks:
        lead_space = t.startswith("Ġ") or t.startswith("▁")
        if t.startswith("Ġ"):
            t = t[1:]
        if t.startswith("▁"):
            t = t[1:]
        t = t.replace("</w>", "")
        out.append((" " if lead_space else "") + t)
    return out


def token_highlight_html(tokens, acts_1d, cfg=CONFIG):
    n = min(len(tokens), len(acts_1d))
    tokens = tokens[:n]
    acts = acts_1d[:n].astype(np.float32)

    if cfg["ignore_special_tokens"]:
        mask = np.ones(n, dtype=bool)
        for i, t in enumerate(tokens):
            if t.strip() in cfg["special_tokens"]:
                mask[i] = False
        mx = float(np.max(acts[mask])) if mask.any() else float(np.max(acts))
    else:
        mx = float(np.max(acts)) if n > 0 else 1.0

    mx = mx if mx > 1e-8 else 1.0
    thr = cfg["min_frac_of_max"] * mx

    parts = []
    for t, a in zip(tokens, acts):
        safe = html_lib.escape(t).replace(" ", "&nbsp;")

        if a <= thr or (
            cfg["ignore_special_tokens"] and t.strip() in cfg["special_tokens"]
        ):
            parts.append(
                f"<span style='font-family:ui-monospace, SFMono-Regular, Menlo, monospace;"
                f"font-size:12px; color:#444; white-space:pre;' title='act={a:.4f}'>"
                f"{safe}</span>"
            )
            continue

        x = float((a - thr) / max(1e-6, (mx - thr)))
        alpha = min(max(x, 0.0), 1.0) * cfg["max_alpha"]
        bg = f"rgba(220,0,0,{alpha:.3f})"
        fg = "#111" if alpha < 0.55 else "#fff"

        parts.append(
            f"<span style='padding:2px 4px; margin:0 0; border-radius:4px; "
            f"background:{bg}; color:{fg}; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; "
            f"font-size:12px; white-space:pre;' title='act={a:.4f}'>"
            f"{safe}</span>"
        )

    return (
        "<div style='line-height:1.8; white-space:pre-wrap;'>"
        + "".join(parts)
        + "</div>"
    )


def flux_grid_hw(height, width):
    return height // 16, width // 16


def make_flux_overlay_patches(image, token_acts_1d, height, width, cfg=CONFIG):
    acts = token_acts_1d.astype(np.float32)

    gh, gw = flux_grid_hw(height, width)
    need = gh * gw

    drop = acts.shape[0] - need
    if drop < 0:
        padded = np.zeros((need,), dtype=np.float32)
        padded[: acts.shape[0]] = acts
        acts = padded
        drop = 0

    grid = acts[drop : drop + need].reshape(gh, gw)

    if np.any(grid > 0):
        p99 = np.percentile(grid[grid > 0], 99)
        grid = np.clip(grid / max(p99, 1e-6), 0, 1)
    else:
        grid = np.zeros_like(grid)

    thr = cfg["min_frac_of_max"]
    strength = np.clip((grid - thr) / max(1e-6, (1.0 - thr)), 0, 1)

    red = np.zeros((gh, gw, 4), dtype=np.uint8)
    red[..., 0] = 220
    red[..., 1] = 0
    red[..., 2] = 0
    red[..., 3] = (strength * (cfg["max_alpha"] * 255)).astype(np.uint8)

    heat_rgba = (
        Image.fromarray(red)
        .resize((width, height), resample=Image.NEAREST)
        .convert("RGBA")
    )
    base = image.resize((width, height)).convert("RGBA")
    overlay = Image.alpha_composite(base, heat_rgba).convert("RGB")

    heat_on_white = Image.new("RGB", (width, height), (255, 255, 255))
    heat_on_white = Image.alpha_composite(
        heat_on_white.convert("RGBA"), heat_rgba
    ).convert("RGB")
    return overlay, heat_on_white


def feature_step_profile(scan, key, feature_idx):
    steps = sorted(scan[key].keys())
    mean = []
    freq = []

    for st in steps:
        tab = scan[key][st]
        total = max(int(tab["total_seen"]), 1)

        m = float(tab["sum_vals"][feature_idx] / total)
        f = float(tab["winner_count"][feature_idx]) / total

        mean.append(m)
        freq.append(f)

    mean = np.array(mean, dtype=np.float32)
    freq = np.array(freq, dtype=np.float32)

    denom = float(mean.max()) if mean.size else 1.0
    mean_norm = mean / max(denom, 1e-8)

    return steps, mean, mean_norm, freq


def plot_feature_vs_step_bars(scan, key, feature_idx, title=None, figsize=(4.6, 1.8)):
    steps, mean, mean_norm, freq = feature_step_profile(scan, key, feature_idx)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    x = np.arange(len(steps))
    ax.bar(x, mean_norm, color="#3b82f6", alpha=0.85, width=0.72)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in steps])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("mean_norm")
    ax.set_xlabel("diffusion step")

    if title:
        ax.set_title(title, fontsize=10)

    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()


def show_feature_header(state, key, feature_idx):
    bs = state.best_step.get(key, {}).get(feature_idx, None)
    display(
        HTML(
            f"""
    <div style="
        margin: 18px 0 10px 0;
        padding: 12px 14px;
        border: 2px solid #111;
        border-left: 10px solid #111;
        background: #f6f6f6;
    ">
      <div style="font-size: 18px; font-weight: 800; line-height: 1.2;">
        {html_lib.escape(key)} · feature {feature_idx}
      </div>
      <div style="margin-top:6px; font-size: 14px; color: #333;">
        best_step: <b>{bs}</b>
      </div>
    </div>
    """
        )
    )


def show_text_feature(
    pipe, prompts, scan, state, key, feature_idx, top_n=5, cfg=CONFIG
):
    assert key.startswith("txt_")
    if feature_idx not in set(state.selected_features.get(key, [])):
        return
    if feature_idx not in state.feat2col[key]:
        return

    show_feature_header(state, key, feature_idx)
    plot_feature_vs_step_bars(
        scan, key, feature_idx, title=f"{key} feature={feature_idx}", figsize=(4.6, 1.8)
    )

    col = state.feat2col[key][feature_idx]
    refs = top_examples_for_feature(scan, key, feature_idx, top_n=top_n)

    for r in refs:
        ex_key = (r.prompt_idx, r.seed)
        if ex_key not in state.example_index:
            continue
        ex_idx = state.example_index[ex_key]

        arr = state.acts[key][r.step][ex_idx]
        acts_1d = arr[:, col]

        prompt = prompts[r.prompt_idx]
        tokens = tokenize_for_display(pipe, prompt)
        html_tokens = token_highlight_html(tokens, acts_1d, cfg=cfg)

        display(
            HTML(
                f"<pre>step={r.step} max={r.value:.4f} seed={r.seed} prompt_idx={r.prompt_idx}</pre>"
                f"<div style='margin:6px 0 8px 0; font-family:ui-sans-serif; color:#111;'>"
                f"<div style='margin-bottom:6px;'><b>prompt:</b> {html_lib.escape(prompt)}</div>"
                f"</div>"
            )
        )
        display(HTML(html_tokens))


def show_image_feature(prompts, scan, state, key, feature_idx, top_n=5, cfg=CONFIG):
    assert key.startswith("img_")
    if feature_idx not in set(state.selected_features.get(key, [])):
        return
    if feature_idx not in state.feat2col[key]:
        return

    show_feature_header(state, key, feature_idx)

    plot_feature_vs_step_bars(
        scan, key, feature_idx, title=f"{key} feature={feature_idx}", figsize=(4.6, 1.8)
    )

    col = state.feat2col[key][feature_idx]
    H, W = state.height, state.width

    refs = top_examples_for_feature(scan, key, feature_idx, top_n=top_n)

    for r in refs:
        ex_key = (r.prompt_idx, r.seed)
        if ex_key not in state.example_index:
            continue
        ex_idx = state.example_index[ex_key]

        arr = state.acts[key][r.step][ex_idx]
        acts_1d = arr[:, col]

        if state.images is None:
            break

        img = state.images[ex_idx].convert("RGB")
        overlay, heat_on_white = make_flux_overlay_patches(img, acts_1d, H, W, cfg=cfg)
        prompt = prompts[r.prompt_idx]

        display(
            HTML(
                f"<pre>step={r.step} max={r.value:.4f} seed={r.seed} prompt_idx={r.prompt_idx}\n"
                f"prompt: {html_lib.escape(prompt)}</pre>"
            )
        )

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img)
        axes[0].set_title("original", fontsize=10)
        axes[0].axis("off")
        axes[1].imshow(overlay)
        axes[1].set_title("overlay", fontsize=10)
        axes[1].axis("off")
        axes[2].imshow(heat_on_white)
        axes[2].set_title("heatmap", fontsize=10)
        axes[2].axis("off")
        plt.tight_layout()
        plt.show()


def pil_to_base64(img, fmt="PNG", max_size=160):
    """Encode a PIL image as a base64 data URI, optionally downscaled."""
    if max_size and (img.width > max_size or img.height > max_size):
        img = img.copy()
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{data}"


def build_feature_tooltips(
    graph_data, pipe, prompts, scan, state, cfg=CONFIG, top_n=None
):
    """Map an attribution graph's feature nodes to per-feature HTML dashboards."""
    top_n = top_n or cfg["top_k_per_feature"]
    H = getattr(state, "height", cfg["height"])
    W = getattr(state, "width", cfg["width"])

    tips = {}
    for nid, node in graph_data["nodes_by_id"].items():
        if node.get("type") != "feature":
            continue
        layer, stream, feat_idx = (
            node.get("layer"),
            node.get("stream"),
            node.get("feat_idx"),
        )
        key = f"{stream}_{layer}"
        if key not in scan or feat_idx not in state.feat2col.get(key, {}):
            continue
        col = state.feat2col[key][feat_idx]
        refs = top_examples_for_feature(scan, key, feat_idx, top_n=top_n)
        if not refs:
            continue

        blocks = [
            "<div style='font-family:sans-serif;font-size:11px;max-width:380px;'>",
            f"<b>{html_lib.escape(key)} f{feat_idx}</b> &middot; top {len(refs)} activations<br>",
        ]
        for r in refs:
            ex_idx = state.example_index.get((r.prompt_idx, r.seed))
            if ex_idx is None:
                continue
            acts_1d = state.acts[key][r.step][ex_idx][:, col]
            blocks.append(
                f"<div style='margin:4px 0;color:#666;'>step={r.step} max={r.value:.3f}</div>"
            )
            if stream == "img" and state.images is not None:
                img = state.images[ex_idx].convert("RGB")
                overlay, _ = make_flux_overlay_patches(img, acts_1d, H, W, cfg=cfg)
                blocks.append(
                    f"<img src='{pil_to_base64(overlay)}' "
                    f"style='border-radius:4px;margin-bottom:4px;'>"
                )
            else:
                tokens = tokenize_for_display(pipe, prompts[r.prompt_idx])
                blocks.append(token_highlight_html(tokens, acts_1d, cfg=cfg))
        blocks.append("</div>")
        tips[nid] = "".join(blocks)
    return tips


@dataclass
class ScanConfig:
    """Configuration for one feature-scanning experiment."""

    name: str
    mode: str  # "contrastive", "spatial", "temporal", "top_activation"

    layers: List[int] = field(default_factory=lambda: list(range(10, 16)))
    streams: List[str] = field(default_factory=lambda: ["img", "txt"])
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    global_top_k: int = 15

    prompt_pairs: Optional[List[Tuple[str, str]]] = None

    prompt: Optional[str] = None
    region_mask: Optional[str] = None
    custom_mask_fn: Optional[Callable] = None

    steps_to_compare: Optional[List[int]] = None


@dataclass
class ScanResult:
    """A single discovered feature with metadata."""

    layer: int
    stream: str
    feat_idx: int
    score: float
    best_prompt: str
    best_seed: int
    best_step: int
    activation_map: Optional[np.ndarray]
    scan_name: str
    per_seed_scores: Dict[int, float]

    def to_target(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "stream": self.stream,
            "feat_idx": self.feat_idx,
            "prompt": self.best_prompt,
            "step": self.best_step,
            "seed": self.best_seed,
        }


class FeatureScanner:
    """Automated feature discovery for circuit tracing experiments."""

    def __init__(self, pipeline: FluxLRMPipeline, device: str = "cuda"):
        self.pipeline = pipeline
        self.device = device
        self._trace_cache: Dict[Tuple, FluxTrace] = {}
        self._image_cache: Dict[Tuple[str, int], Image.Image] = {}

    def scan_and_trace(
        self,
        config: ScanConfig,
        output_dir: str,
    ) -> Optional[pd.DataFrame]:
        os.makedirs(output_dir, exist_ok=True)

        candidates = self._scan(config)
        selected = sorted(candidates, key=lambda c: c["score"], reverse=True)
        selected = selected[: config.global_top_k]
        for i, c in enumerate(selected):
            n_seeds = len(c["per_seed_scores"])
            print(
                f"    {i+1:2d}. L{c['layer']:2d}/{c['stream']}  "
                f"f{c['feat_idx']:5d}  score={c['score']:.4f}  "
                f"({n_seeds}/{len(config.seeds)} seeds)"
            )

        results = self._assign_best_step(selected, config)
        self._visualize_in_notebook(results, config)

        targets = [r.to_target() for r in results]
        self.clear_cache()

        if not targets:
            return None

        df = self.pipeline.run_experiment(
            targets,
            output_dir=f"{output_dir}/circuits",
            perturbation_top_k=30,
        )
        df.to_csv(f"{output_dir}/circuit_summary.csv", index=False)
        return df

    def clear_cache(self):
        self._trace_cache.clear()
        self._image_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def _scan(self, config: ScanConfig) -> List[Dict]:
        dispatch = {
            "contrastive": self._scan_contrastive,
            "spatial": self._scan_spatial,
            "temporal": self._scan_temporal,
            "top_activation": self._scan_top_activation,
        }
        if config.mode not in dispatch:
            raise ValueError(f"Unknown mode: {config.mode!r}")
        return dispatch[config.mode](config)

    def _scan_contrastive(self, config: ScanConfig) -> List[Dict]:
        assert config.prompt_pairs, "Contrastive mode requires prompt_pairs"
        D = self.pipeline.cfg.expansion_factor * self.pipeline.cfg.d_model

        per_seed_scores: Dict[int, Dict[tuple, float]] = {}

        for seed in config.seeds:
            pair_deltas: Dict[Tuple[int, str], List[Tensor]] = {}

            for prompt_a, prompt_b in config.prompt_pairs:
                trace_a = self._get_trace(prompt_a, seed, 0)
                trace_b = self._get_trace(prompt_b, seed, 0)

                for layer in config.layers:
                    for stream in config.streams:
                        z_a = self._get_z(trace_a, layer, stream)
                        z_b = self._get_z(trace_b, layer, stream)
                        if z_a is None or z_b is None:
                            continue

                        max_a = z_a[0].to(self.device).max(dim=0).values
                        max_b = z_b[0].to(self.device).max(dim=0).values
                        delta = (max_a - max_b).cpu()

                        key = (layer, stream)
                        if key not in pair_deltas:
                            pair_deltas[key] = []
                        pair_deltas[key].append(delta)

            seed_scores: Dict[tuple, float] = {}
            for (layer, stream), deltas in pair_deltas.items():
                stacked = torch.stack(deltas, dim=0)
                avg_abs = stacked.abs().mean(dim=0)
                sign_cons = stacked.sign().mean(dim=0).abs()
                final = avg_abs * sign_cons

                top_k = min(200, D)
                top_vals, top_idxs = final.topk(top_k)
                for rank in range(top_k):
                    feat_idx = top_idxs[rank].item()
                    s = top_vals[rank].item()
                    if s > 1e-6:
                        seed_scores[(layer, stream, feat_idx)] = s

            per_seed_scores[seed] = seed_scores
            self._trace_cache.clear()
            gc.collect()
            torch.cuda.empty_cache()

        return self._aggregate_seeds(per_seed_scores, config.seeds)

    def _scan_spatial(self, config: ScanConfig) -> List[Dict]:
        assert config.prompt, "Spatial mode requires prompt"
        per_seed_scores: Dict[int, Dict[tuple, float]] = {}

        for seed in config.seeds:
            trace = self._get_trace(config.prompt, seed, 0)
            mask = self._make_region_mask(config, trace.S_img).to(self.device)

            seed_scores: Dict[tuple, float] = {}
            for layer in config.layers:
                z = self._get_z(trace, layer, "img")
                if z is None:
                    continue
                z_dev = z[0].to(self.device)
                mean_in = z_dev[mask].mean(dim=0)
                mean_out = z_dev[~mask].mean(dim=0)
                score = (mean_in - mean_out).cpu()

                top_k = min(200, score.shape[0])
                top_vals, top_idxs = score.topk(top_k)
                for rank in range(top_k):
                    feat_idx = top_idxs[rank].item()
                    s = top_vals[rank].item()
                    if s > 1e-6:
                        seed_scores[(layer, "img", feat_idx)] = s

            per_seed_scores[seed] = seed_scores
            self._trace_cache.clear()
            gc.collect()
            torch.cuda.empty_cache()

        return self._aggregate_seeds(per_seed_scores, config.seeds)

    def _scan_temporal(self, config: ScanConfig) -> List[Dict]:
        assert config.prompt, "Temporal mode requires prompt"
        assert config.steps_to_compare and len(config.steps_to_compare) >= 2
        per_seed_scores: Dict[int, Dict[tuple, float]] = {}

        for seed in config.seeds:
            traces = {
                s: self._get_trace(config.prompt, seed, s)
                for s in config.steps_to_compare
            }
            seed_scores: Dict[tuple, float] = {}
            for layer in config.layers:
                for stream in config.streams:
                    step_maxes = []
                    for step in config.steps_to_compare:
                        z = self._get_z(traces[step], layer, stream)
                        if z is None:
                            break
                        step_maxes.append(z[0].to(self.device).max(dim=0).values)

                    if len(step_maxes) != len(config.steps_to_compare):
                        continue

                    stacked = torch.stack(step_maxes, dim=0)
                    variance = stacked.var(dim=0).cpu()

                    top_k = min(200, variance.shape[0])
                    top_vals, top_idxs = variance.topk(top_k)
                    for rank in range(top_k):
                        feat_idx = top_idxs[rank].item()
                        s = top_vals[rank].item()
                        if s > 1e-6:
                            seed_scores[(layer, stream, feat_idx)] = s

            per_seed_scores[seed] = seed_scores
            self._trace_cache.clear()
            gc.collect()
            torch.cuda.empty_cache()

        return self._aggregate_seeds(per_seed_scores, config.seeds)

    def _scan_top_activation(self, config: ScanConfig) -> List[Dict]:
        assert config.prompt, "top_activation mode requires prompt"
        per_seed_scores: Dict[int, Dict[tuple, float]] = {}

        for seed in config.seeds:
            trace = self._get_trace(config.prompt, seed, 0)
            seed_scores: Dict[tuple, float] = {}
            for layer in config.layers:
                for stream in config.streams:
                    z = self._get_z(trace, layer, stream)
                    if z is None:
                        continue
                    max_act = z[0].to(self.device).max(dim=0).values.cpu()

                    top_k = min(200, max_act.shape[0])
                    top_vals, top_idxs = max_act.topk(top_k)
                    for rank in range(top_k):
                        feat_idx = top_idxs[rank].item()
                        s = top_vals[rank].item()
                        if s > 1e-6:
                            seed_scores[(layer, stream, feat_idx)] = s

            per_seed_scores[seed] = seed_scores
            self._trace_cache.clear()
            gc.collect()
            torch.cuda.empty_cache()

        return self._aggregate_seeds(per_seed_scores, config.seeds)

    def _aggregate_seeds(self, per_seed_scores, seeds):
        all_keys: Set[tuple] = set()
        for ss in per_seed_scores.values():
            all_keys.update(ss.keys())

        aggregated = []
        for key in all_keys:
            layer, stream, feat_idx = key
            seed_vals = {}
            for seed in seeds:
                if key in per_seed_scores.get(seed, {}):
                    seed_vals[seed] = per_seed_scores[seed][key]
            if not seed_vals:
                continue
            avg_score = sum(seed_vals.values()) / len(seeds)
            best_seed = max(seed_vals, key=seed_vals.get)
            aggregated.append(
                {
                    "layer": layer,
                    "stream": stream,
                    "feat_idx": feat_idx,
                    "score": avg_score,
                    "best_seed": best_seed,
                    "per_seed_scores": seed_vals,
                }
            )
        return aggregated

    def _assign_best_step(self, selected, config):
        steps = config.steps_to_compare or list(
            range(self.pipeline.cfg.num_inference_steps)
        )
        results = []

        for cand in selected:
            layer = cand["layer"]
            stream = cand["stream"]
            feat_idx = cand["feat_idx"]
            seed = cand["best_seed"]
            prompt = self._get_best_prompt(cand, config)

            best_step = 0
            best_act = -1.0
            best_act_map = None

            for step in steps:
                trace = self._get_trace(prompt, seed, step)
                z = self._get_z(trace, layer, stream)
                if z is None:
                    continue
                acts = z[0, :, feat_idx].cpu()
                max_act = float(acts.max().item())
                if max_act > best_act:
                    best_act = max_act
                    best_step = step
                    best_act_map = acts.numpy()

            results.append(
                ScanResult(
                    layer=layer,
                    stream=stream,
                    feat_idx=feat_idx,
                    score=cand["score"],
                    best_prompt=prompt,
                    best_seed=seed,
                    best_step=best_step,
                    activation_map=best_act_map,
                    scan_name=config.name,
                    per_seed_scores=cand["per_seed_scores"],
                )
            )

        return results

    def _visualize_in_notebook(self, selected, config):
        n = len(selected)
        if n == 0:
            return

        cols = min(5, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
        if n == 1:
            axes = np.array([[axes]])
        else:
            axes = np.atleast_2d(axes)
            if axes.shape[0] == 1 and rows > 1:
                axes = axes.reshape(rows, cols)

        for i, r in enumerate(selected):
            row, col = divmod(i, cols)
            ax = axes[row][col] if rows > 1 else axes[0][col]

            img = self._get_image(r.best_prompt, r.best_seed)

            if r.stream == "img" and r.activation_map is not None:
                overlay, _ = make_flux_overlay_patches(
                    img,
                    r.activation_map,
                    self.pipeline.cfg.height,
                    self.pipeline.cfg.width,
                )
                ax.imshow(overlay)
            else:
                ax.imshow(img)
                ax.text(
                    0.5,
                    0.02,
                    "[txt stream]",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=7,
                    color="white",
                    bbox=dict(boxstyle="round", fc="black", alpha=0.7),
                )

            n_seeds = len(r.per_seed_scores)
            ax.set_title(
                f"L{r.layer}/{r.stream} f{r.feat_idx}\n"
                f"score={r.score:.3f}  step={r.best_step}  "
                f"seed={r.best_seed} ({n_seeds}/{len(config.seeds)})",
                fontsize=7,
            )
            ax.axis("off")

        for i in range(n, rows * cols):
            row, col = divmod(i, cols)
            ax = axes[row][col] if rows > 1 else axes[0][col]
            ax.axis("off")

        fig.suptitle(f"{config.name} - Top {n} features", fontsize=12, y=1.01)
        plt.tight_layout()
        plt.show()

    def _get_trace(self, prompt, seed, step):
        key = (prompt, step, seed)
        if key not in self._trace_cache:
            self._trace_cache[key] = self.pipeline.capture(prompt, seed, step)
        return self._trace_cache[key]

    def _get_z(self, trace, layer, stream):
        lc = trace.get_layer(layer)
        return getattr(lc, f"{stream}_z", None)

    def _get_image(self, prompt, seed):
        key = (prompt, seed)
        if key not in self._image_cache:
            cfg = self.pipeline.cfg
            gen = torch.Generator(cfg.device).manual_seed(seed)
            result = self.pipeline.pipe(
                prompt,
                prompt_2=prompt,
                height=cfg.height,
                width=cfg.width,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                generator=gen,
                output_type="pil",
            )
            self._image_cache[key] = result.images[0]
        return self._image_cache[key]

    def _get_best_prompt(self, cand, config):
        if config.mode == "contrastive" and config.prompt_pairs:
            return config.prompt_pairs[0][0]
        return config.prompt or ""

    def _make_region_mask(self, config, S_img):
        H = W = int(S_img**0.5)
        assert H * W == S_img, f"S_img={S_img} is not a perfect square"

        if config.custom_mask_fn is not None:
            return config.custom_mask_fn(H, W)

        region = config.region_mask
        assert region, "Spatial mode requires region_mask or custom_mask_fn"

        mask_2d = torch.zeros(H, W, dtype=torch.bool)
        if region == "left_half":
            mask_2d[:, : W // 2] = True
        elif region == "right_half":
            mask_2d[:, W // 2 :] = True
        elif region == "top_half":
            mask_2d[: H // 2, :] = True
        elif region == "bottom_half":
            mask_2d[H // 2 :, :] = True
        elif region == "center":
            h4, w4 = H // 4, W // 4
            mask_2d[h4 : H - h4, w4 : W - w4] = True
        else:
            raise ValueError(f"Unknown region: {region!r}")
        return mask_2d.flatten()
