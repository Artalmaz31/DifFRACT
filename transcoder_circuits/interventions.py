import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from diffusers.models.embeddings import get_timestep_embedding
from IPython.display import display
import logging
import matplotlib

matplotlib.use("Agg")
logging.getLogger("diffusers").setLevel(logging.ERROR)


@dataclass
class FeatureIntervention:
    """Specification for modifying a single transcoder feature during generation."""

    layer: int
    stream: str
    feat_idx: int
    mode: str  # "amplify", "zero", "invert", "set"
    scale: float = 1.0
    steps: Optional[Set[int]] = None
    token_mode: str = "all"
    token_threshold: float = 0.20


@dataclass
class StreamIntervention:
    """Zero out all features in a stream at specified layers and steps."""

    layers: List[int]
    stream: str
    steps: Set[int]


@dataclass
class InterventionConfig:
    """Full specification for one intervention experiment run."""

    name: str
    prompt: str
    seeds: List[int]
    interventions: List[FeatureIntervention] = field(default_factory=list)
    stream_interventions: List[StreamIntervention] = field(default_factory=list)
    description: str = ""


# A supernode is a list of (layer, stream, feat_idx) tuples representing one
# semantic concept. A concept is usually distributed across several features, so
# to steer it effectively every feature in the supernode is acted on together
Supernode = List[Tuple[int, str, int]]


@dataclass
class SteerSweepConfig:
    """Generates a grid of InterventionConfigs by sweeping scale values."""

    name: str
    prompt: str
    seeds: List[int]
    supernode: Supernode
    scales: List[float]
    stream_interventions: List[StreamIntervention] = field(default_factory=list)
    description: str = ""
    mode: str = "amplify"


def expand_sweep(sweep: SteerSweepConfig) -> List[InterventionConfig]:
    configs = []
    for scale in sweep.scales:
        interventions = [
            FeatureIntervention(
                layer=l,
                stream=s,
                feat_idx=f,
                mode=sweep.mode,
                scale=scale,
            )
            for (l, s, f) in sweep.supernode
        ]
        configs.append(
            InterventionConfig(
                name=f"{sweep.name}_scale{scale:+g}",
                prompt=sweep.prompt,
                seeds=sweep.seeds,
                interventions=interventions,
                stream_interventions=sweep.stream_interventions,
                description=f"{sweep.description} (scale={scale})",
            )
        )
    return configs


class _MultiReplaceFF(nn.Module):
    def __init__(self, controller, key, stream):
        super().__init__()
        self.controller = controller
        self.key = key
        self.stream = stream

    def forward(self, x, *args, **kwargs):
        return self.controller.forward_transcoder_ff(self.key, self.stream, x)


class MultiFeatureController:
    """Context manager that replaces MLP layers with transcoder-based computation,
    applying multiple feature interventions across multiple layers simultaneously."""

    FEAT_CHUNK = 4096  # Process features in chunks to fit in memory

    def __init__(
        self,
        pipe,
        transcoders: Dict[str, nn.Module],
        interventions: List[FeatureIntervention],
        stream_interventions: Optional[List[StreamIntervention]] = None,
        height: int = 512,
        width: int = 512,
    ):
        self.pipe = pipe
        self.transcoders = transcoders
        self.interventions = interventions
        self.stream_interventions = stream_interventions or []
        self.height = height
        self.width = width

        self.step_call_idx = -1
        self.current_timestep = None
        self.hooks = []
        self.originals: Dict[str, Any] = {}

        self._grouped: Dict[str, List[FeatureIntervention]] = {}
        for intv in self.interventions:
            key = f"{intv.stream}_{intv.layer}"
            self._grouped.setdefault(key, []).append(intv)

        self._stream_zero: Dict[str, Set[int]] = {}
        for si in self.stream_interventions:
            for layer in si.layers:
                key = f"{si.stream}_{layer}"
                self._stream_zero.setdefault(key, set()).update(si.steps)

        self._all_keys = set(self._grouped.keys()) | set(self._stream_zero.keys())

    def _on_transformer_pre(self, module, args, kwargs):
        self.step_call_idx += 1
        self.current_timestep = kwargs.get("timestep", None)
        return None

    @torch.no_grad()
    def forward_transcoder_ff(self, key, stream, x):
        tc = self.transcoders[key]
        tc_on_cpu = next(tc.parameters()).device.type == "cpu"
        if tc_on_cpu:
            tc.to(x.device)
        t = self.current_timestep
        step = self.step_call_idx

        B, S, D = x.shape
        Fdim = tc.d_feat

        # Check stream-level zero
        stream_zero_this_step = (
            key in self._stream_zero and step in self._stream_zero[key]
        )

        # Get feature interventions active this step
        active_intvs = []
        for intv in self._grouped.get(key, []):
            if intv.steps is None or step in intv.steps:
                active_intvs.append(intv)

        # If no interventions use original computation
        if not stream_zero_this_step and not active_intvs:
            rec, z = tc(x.to(dtype=tc.encoder.weight.dtype), t)
            if tc_on_cpu:
                tc.cpu()
            return rec.to(dtype=x.dtype)

        # Modulate input
        x_enc = x.to(dtype=tc.encoder.weight.dtype)
        t_float = t.to(dtype=torch.float32, device=x.device).view(-1)
        t_emb = get_timestep_embedding(t_float, embedding_dim=tc.time_embed_dim).to(
            dtype=x_enc.dtype, device=x_enc.device
        )
        scale, shift = tc.get_modulation(t_emb)
        x_mod = x_enc * (1.0 + scale[:, None, :]) + shift[:, None, :]

        # Build output from decoder bias
        y = tc.decoder.bias[None, None, :].to(
            device=x.device, dtype=tc.decoder.weight.dtype
        )
        y = y.expand(B, S, D).clone()

        # Process in chunks
        for i in range(0, Fdim, self.FEAT_CHUNK):
            j = min(i + self.FEAT_CHUNK, Fdim)

            Wenc = tc.encoder.weight[i:j]
            benc = tc.encoder.bias[i:j]
            z = F.relu(F.linear(x_mod, Wenc, benc))

            if stream_zero_this_step:
                z.zero_()
            else:
                # Apply feature-level interventions
                for intv in active_intvs:
                    f = intv.feat_idx
                    if i <= f < j:
                        pos = f - i
                        zf = z[:, :, pos]

                        if intv.token_mode == "threshold":
                            zmax = zf.amax(dim=1, keepdim=True).clamp(min=1e-6)
                            mask = zf > (zmax * intv.token_threshold)
                        else:
                            mask = torch.ones_like(zf, dtype=torch.bool)

                        if intv.mode == "zero":
                            zf_new = torch.zeros_like(zf)
                        elif intv.mode == "amplify":
                            zf_new = zf * intv.scale
                        elif intv.mode == "invert":
                            zf_new = zf * -1.0
                        elif intv.mode == "set":
                            zf_new = torch.full_like(zf, intv.scale)
                        else:
                            raise ValueError(f"Unknown mode: {intv.mode}")

                        z[:, :, pos] = torch.where(mask, zf_new, zf)

            Wdec = tc.decoder.weight[:, i:j]
            y = y + F.linear(z, Wdec, None)

        if tc_on_cpu:
            tc.cpu()

        return y.to(dtype=x.dtype)

    def __enter__(self):
        self.step_call_idx = -1
        self.current_timestep = None
        self.originals = {}

        # Install step-tracking hook
        self.hooks.append(
            self.pipe.transformer.register_forward_pre_hook(
                self._on_transformer_pre, with_kwargs=True
            )
        )

        # Patch each (layer, stream) that has interventions
        for key in self._all_keys:
            parts = key.split("_")
            stream = parts[0]
            layer = int(parts[1])

            if key not in self.transcoders:
                print(f"  WARNING: no transcoder for {key}, skipping")
                continue

            blk = self.pipe.transformer.transformer_blocks[layer]
            replacement = _MultiReplaceFF(self, key=key, stream=stream)

            if stream == "img":
                self.originals[key] = blk.ff
                blk.ff = replacement
            elif stream == "txt":
                self.originals[key] = blk.ff_context
                blk.ff_context = replacement

        return self

    def __exit__(self, exc_type, exc, tb):
        # Restore original modules
        for key, orig in self.originals.items():
            parts = key.split("_")
            stream = parts[0]
            layer = int(parts[1])
            blk = self.pipe.transformer.transformer_blocks[layer]
            if stream == "img":
                blk.ff = orig
            elif stream == "txt":
                blk.ff_context = orig

        # Remove hooks
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.originals.clear()

        return False


class InterventionRunner:
    """Runs intervention experiments."""

    def __init__(
        self,
        pipe,
        transcoders: Dict[str, nn.Module],
        device: str = "cuda",
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 4,
    ):
        self.pipe = pipe
        self.transcoders = transcoders
        self.device = device
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps

    @torch.inference_mode()
    def _generate(self, prompt: str, seed: int) -> Image.Image:
        gen = torch.Generator(device=self.device).manual_seed(seed)
        out = self.pipe(
            prompt,
            prompt_2=prompt,
            height=self.height,
            width=self.width,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=0.0,
            output_type="pil",
            generator=gen,
        )
        return out.images[0]

    @torch.inference_mode()
    def _generate_with_intervention(
        self,
        prompt: str,
        seed: int,
        interventions: List[FeatureIntervention],
        stream_interventions: List[StreamIntervention],
    ) -> Image.Image:
        with MultiFeatureController(
            pipe=self.pipe,
            transcoders=self.transcoders,
            interventions=interventions,
            stream_interventions=stream_interventions,
            height=self.height,
            width=self.width,
        ):
            gen = torch.Generator(device=self.device).manual_seed(seed)
            out = self.pipe(
                prompt,
                prompt_2=prompt,
                height=self.height,
                width=self.width,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=0.0,
                output_type="pil",
                generator=gen,
            )
        return out.images[0]

    def run_single(self, config: InterventionConfig, seed: int) -> Dict[str, Any]:
        baseline = self._generate(config.prompt, seed)
        intervention = self._generate_with_intervention(
            config.prompt,
            seed,
            config.interventions,
            config.stream_interventions,
        )
        return {
            "seed": seed,
            "prompt": config.prompt,
            "config_name": config.name,
            "baseline": baseline,
            "intervention": intervention,
        }

    def run_batch(self, config: InterventionConfig) -> List[Dict[str, Any]]:
        results = []
        for i, seed in enumerate(config.seeds):
            print(f"    seed {seed} ({i+1}/{len(config.seeds)})")
            results.append(self.run_single(config, seed))
        return results

    def run_experiment(
        self,
        configs: List[InterventionConfig],
        output_dir: str,
    ) -> List[Dict[str, Any]]:
        os.makedirs(output_dir, exist_ok=True)
        all_results = []
        for config in configs:
            print(f"{config.name}: {len(config.seeds)} seeds | {config.prompt!r}")
            config_dir = os.path.join(output_dir, config.name)
            os.makedirs(config_dir, exist_ok=True)
            results = self.run_batch(config)
            for r in results:
                seed = r["seed"]
                r["baseline"].save(
                    os.path.join(config_dir, f"seed_{seed}_baseline.png")
                )
                r["intervention"].save(
                    os.path.join(config_dir, f"seed_{seed}_intervention.png")
                )
            grid = make_comparison_grid(results, config.name, config.description)
            grid.save(os.path.join(config_dir, "comparison_grid.png"))
            display(grid)
            all_results.extend(results)
        return all_results

    def run_sweep(
        self,
        sweep: SteerSweepConfig,
        output_dir: str,
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        sweep_dir = os.path.join(output_dir, sweep.name)
        os.makedirs(sweep_dir, exist_ok=True)

        print(
            f"{sweep.name}: {len(sweep.supernode)} feats x "
            f"{len(sweep.scales)} scales x {len(sweep.seeds)} seeds | {sweep.prompt!r}"
        )
        baselines = {}
        for seed in sweep.seeds:
            baselines[seed] = self._generate(sweep.prompt, seed)

        sweep_images = {}
        for scale_i, scale in enumerate(sweep.scales):
            print(f"  scale {scale:+g} ({scale_i + 1}/{len(sweep.scales)})")
            interventions = [
                FeatureIntervention(
                    layer=l,
                    stream=s,
                    feat_idx=f,
                    mode=sweep.mode,
                    scale=scale,
                )
                for (l, s, f) in sweep.supernode
            ]
            for seed in sweep.seeds:
                img = self._generate_with_intervention(
                    sweep.prompt,
                    seed,
                    interventions,
                    sweep.stream_interventions,
                )
                sweep_images[(scale, seed)] = img

        for seed, img in baselines.items():
            img.save(os.path.join(sweep_dir, f"baseline_seed{seed}.png"))
        for (scale, seed), img in sweep_images.items():
            img.save(os.path.join(sweep_dir, f"scale{scale:+g}_seed{seed}.png"))

        collage = make_sweep_collage(
            baselines=baselines,
            sweep_images=sweep_images,
            scales=sweep.scales,
            seeds=sweep.seeds,
            title=sweep.name,
            description=sweep.description,
        )
        collage.save(os.path.join(sweep_dir, "sweep_collage.png"))
        display(collage)

        return {
            "sweep_name": sweep.name,
            "baselines": baselines,
            "sweep_images": sweep_images,
            "output_dir": sweep_dir,
        }

    def run_sweeps(
        self,
        sweeps: List[SteerSweepConfig],
        output_dir: str,
    ) -> List[Dict[str, Any]]:
        os.makedirs(output_dir, exist_ok=True)
        results = []
        for sweep in sweeps:
            try:
                result = self.run_sweep(sweep, output_dir)
                results.append(result)
            except Exception as e:
                import traceback

                print(f"  FAILED: {sweep.name}: {e}")
                traceback.print_exc()
        return results

    def run_batch_prompts(
        self,
        prompts_and_meta: List[Tuple[str, Any]],
        intervention_builder,
        seed: int,
        name: str,
        output_dir: str,
    ) -> List[Dict[str, Any]]:
        batch_dir = os.path.join(output_dir, name)
        os.makedirs(batch_dir, exist_ok=True)

        print(f"{name}: {len(prompts_and_meta)} prompts")

        results = []
        for i, (prompt, meta) in enumerate(prompts_and_meta):
            interventions = intervention_builder(meta)
            if not interventions:
                print(
                    f"    [{i+1}/{len(prompts_and_meta)}] {prompt!r} SKIP (no valid features)"
                )
                continue
            print(
                f"    [{i+1}/{len(prompts_and_meta)}] {prompt!r} ({len(interventions)} intv)"
            )
            baseline = self._generate(prompt, seed)
            intervention = self._generate_with_intervention(
                prompt,
                seed,
                interventions,
                [],
            )
            results.append(
                {
                    "seed": seed,
                    "prompt": prompt,
                    "meta": meta,
                    "baseline": baseline,
                    "intervention": intervention,
                    "config_name": name,
                }
            )

        if results:
            grid = make_comparison_grid(
                results, name, f"batch ({len(results)} prompts)"
            )
            grid.save(os.path.join(batch_dir, "batch_collage.png"))
            display(grid)
            for r in results:
                slug = r["prompt"].replace(" ", "_").replace(",", "")[:40]
                r["baseline"].save(os.path.join(batch_dir, f"{slug}_baseline.png"))
                r["intervention"].save(
                    os.path.join(batch_dir, f"{slug}_intervention.png")
                )

        return results


def _get_font(size=16):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            return ImageFont.load_default()


def make_comparison_grid(
    results: List[Dict[str, Any]],
    title: str = "",
    description: str = "",
    cols: int = 5,
) -> Image.Image:
    n = len(results)
    if n == 0:
        return Image.new("RGB", (100, 100), "white")

    cols = min(cols, n)
    rows_of_pairs = (n + cols - 1) // cols

    w, h = results[0]["baseline"].size
    pad = 4
    label_h = 24
    title_h = 40 if title else 0
    left_label_w = 100

    grid_w = left_label_w + cols * (w + pad) + pad
    grid_h = title_h + rows_of_pairs * (2 * h + 2 * label_h + pad) + pad

    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)
    font = _get_font(14)
    title_font = _get_font(16)

    if title:
        desc_text = f"{title}: {description}" if description else title
        draw.text((pad, 8), desc_text[:200], fill="black", font=title_font)

    for idx, r in enumerate(results):
        col = idx % cols
        row_pair = idx // cols
        x = left_label_w + pad + col * (w + pad)
        y_base = title_h + row_pair * (2 * h + 2 * label_h + pad) + pad
        if col == 0:
            draw.text((pad, y_base + h // 2 - 10), "baseline", fill="black", font=font)
            draw.text(
                (pad, y_base + h + label_h + h // 2 - 10),
                "intervention",
                fill="blue",
                font=font,
            )

        grid.paste(r["baseline"], (x, y_base))
        grid.paste(r["intervention"], (x, y_base + h + label_h))

        seed = r["seed"]
        label = f"seed={seed}"
        draw.text((x, y_base + 2 * h + label_h + 2), label, fill="black", font=font)

        prompt = r.get("prompt", "")
        if prompt:
            prompt_short = prompt[:35]
            draw.text((x, y_base + h - 18), prompt_short, fill="gray", font=font)

    return grid


def make_sweep_collage(
    baselines: Dict[int, Image.Image],
    sweep_images: Dict[Tuple[float, int], Image.Image],
    scales: List[float],
    seeds: List[int],
    title: str = "",
    description: str = "",
) -> Image.Image:
    if not baselines or not sweep_images:
        return Image.new("RGB", (100, 100), "white")

    first_seed = seeds[0]
    w, h = baselines[first_seed].size

    pad = 4
    label_w = 120
    header_h = 30
    title_h = 40 if title else 0

    n_cols = len(seeds)
    n_rows = 1 + len(scales)

    collage_w = label_w + n_cols * (w + pad) + pad
    collage_h = title_h + header_h + n_rows * (h + pad) + pad

    collage = Image.new("RGB", (collage_w, collage_h), "white")
    draw = ImageDraw.Draw(collage)
    font = _get_font(14)
    title_font = _get_font(18)

    if title:
        desc_text = f"{title}: {description}" if description else title
        draw.text((pad, 10), desc_text[:200], fill="black", font=title_font)

    for c, seed in enumerate(seeds):
        x = label_w + pad + c * (w + pad)
        draw.text(
            (x + w // 2 - 25, title_h + 5), f"seed={seed}", fill="black", font=font
        )

    y = title_h + header_h
    draw.text((5, y + h // 2 - 10), "BASELINE", fill="darkgreen", font=font)
    for c, seed in enumerate(seeds):
        x = label_w + pad + c * (w + pad)
        if seed in baselines:
            collage.paste(baselines[seed], (x, y))

    for r, scale in enumerate(scales):
        y = title_h + header_h + (r + 1) * (h + pad)
        draw.text((5, y + h // 2 - 10), f"scale={scale:+g}", fill="darkred", font=font)
        for c, seed in enumerate(seeds):
            x = label_w + pad + c * (w + pad)
            if (scale, seed) in sweep_images:
                collage.paste(sweep_images[(scale, seed)], (x, y))

    return collage
