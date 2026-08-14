import gc
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from PIL import Image, ImageDraw


class TranscoderInferenceWrapper(nn.Module):
    """Drop-in replacement for a block.ff that runs the transcoder instead of the MLP."""

    def __init__(self, transcoder, orig_dtype, t_ctx):
        super().__init__()
        self.tc = transcoder
        self.orig_dtype = orig_dtype
        self.t_ctx = t_ctx

    def forward(self, x, *args, **kwargs):
        B, S, D = x.shape
        inp = x.reshape(B * S, D).to(self.tc.encoder.weight.dtype)

        t = self.t_ctx.t.to(inp.device, dtype=torch.float32)
        t = t.repeat_interleave(S)

        rec, _ = self.tc(inp, t)
        rec = rec.reshape(B, S, D)

        return rec.to(self.orig_dtype)


class SAEInferenceWrapper(nn.Module):
    """Drop-in replacement for a block.ff that runs SAE(MLP(x))."""

    def __init__(self, sae, orig_ff, orig_dtype, t_ctx):
        super().__init__()
        self.sae = sae
        self.orig_ff = orig_ff
        self.orig_dtype = orig_dtype
        self.t_ctx = t_ctx

    def forward(self, x, *args, **kwargs):
        y = self.orig_ff(x, *args, **kwargs)
        B, S, D = y.shape
        inp = y.reshape(B * S, D).to(self.sae.encoder.weight.dtype)

        t = self.t_ctx.t.to(inp.device, dtype=torch.float32)
        t = t.repeat_interleave(S)

        rec, _ = self.sae(inp, t)
        return rec.reshape(B, S, D).to(self.orig_dtype)


def _unpack_flux_latents(latents, height, width):
    batch_size, num_patches, channels = latents.shape
    h_latent = height // 8
    w_latent = width // 8
    latents = latents.view(
        batch_size, h_latent // 2, w_latent // 2, channels // 4, 2, 2
    )
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // 4, h_latent, w_latent)
    return latents


def _decode_latent_to_pil(pipe, latent_tensor, height, width, device):
    lat = latent_tensor.unsqueeze(0).to(device).type(pipe.vae.dtype)
    if lat.ndim == 3:
        lat = _unpack_flux_latents(lat, height, width)
    shift = getattr(pipe.vae.config, "shift_factor", 0.0)
    lat = lat / pipe.vae.config.scaling_factor + shift
    with torch.inference_mode():
        image = pipe.vae.decode(lat).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return Image.fromarray((image[0] * 255).round().astype("uint8"))


@torch.no_grad()
def run_validation(
    pipe,
    models: nn.ModuleDict,
    keys: Sequence[Tuple[str, int]],
    prompts: List[str],
    capturer,
    t_ctx,
    *,
    kind: str = "transcoder",
    num_inference_steps: int = 4,
    batch_size: int = 16,
    device: str = "cuda",
    height: int = 512,
    width: int = 512,
    orig_dtype: torch.dtype = torch.bfloat16,
    guidance_scale: float = 0.0,
    prompt_aliases: Sequence[str] = ("prompt_2",),
    make_comparison_image: bool = True,
) -> Tuple[float, float, Optional["Image.Image"]]:
    """Replace all target MLPs with models and compare final latents to the original."""
    torch.cuda.empty_cache()
    gc.collect()
    models.eval()

    prev_enabled = capturer.enabled
    capturer.enabled = False

    blocks = pipe.transformer.transformer_blocks
    backup_layers: Dict[str, nn.Module] = {}
    repl_layers: Dict[str, nn.Module] = {}
    for stream, l in keys:
        key = f"{stream}_{l}"
        orig = blocks[l].ff if stream == "img" else blocks[l].ff_context
        backup_layers[key] = orig
        model = models[key]
        if kind == "sae":
            repl_layers[key] = SAEInferenceWrapper(model, orig, orig_dtype, t_ctx)
        else:
            repl_layers[key] = TranscoderInferenceWrapper(model, orig_dtype, t_ctx)

    def _assign(table: Dict[str, nn.Module]):
        for stream, l in keys:
            blk = blocks[l]
            module = table[f"{stream}_{l}"]
            if stream == "img":
                blk.ff = module
            else:
                blk.ff_context = module

    def set_model_to_original():
        _assign(backup_layers)

    def set_model_to_replacement():
        _assign(repl_layers)

    mse_accum = cos_accum = 0.0
    valid_count = 0
    viz_orig = viz_repl = None

    pbar = tqdm(range(0, len(prompts), batch_size), desc="Validation", leave=False)
    for i in pbar:
        batch_prompts = prompts[i : i + batch_size]
        cur = len(batch_prompts)

        def _gen():
            return [
                torch.Generator(device=device).manual_seed(2025 + (i + j))
                for j in range(cur)
            ]

        def _call():
            kwargs = {alias: batch_prompts for alias in prompt_aliases}
            return pipe(
                batch_prompts,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                output_type="latent",
                generator=_gen(),
                **kwargs,
            ).images.cpu()

        set_model_to_original()

        with torch.inference_mode():
            lat_orig = _call()

        if i == 0:
            viz_orig = lat_orig[0].clone()

        set_model_to_replacement()

        with torch.inference_mode():
            lat_repl = _call()

        if i == 0:
            viz_repl = lat_repl[0].clone()

        for j in range(cur):
            a, b = lat_orig[j], lat_repl[j]
            mse_accum += F.mse_loss(a.float(), b.float()).item()
            cos_accum += F.cosine_similarity(
                a.view(1, -1).float(), b.view(1, -1).float()
            ).item()
            valid_count += 1

        del lat_orig, lat_repl
        gc.collect()

    set_model_to_original()
    capturer.enabled = prev_enabled

    repl_layers.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    comparison_image = None
    if make_comparison_image and viz_orig is not None and viz_repl is not None:
        img_orig = _decode_latent_to_pil(pipe, viz_orig, height, width, device)
        img_repl = _decode_latent_to_pil(pipe, viz_repl, height, width, device)
        w_img, h_img = img_orig.size
        comparison_image = Image.new("RGB", (w_img * 2, h_img + 30), (255, 255, 255))
        comparison_image.paste(img_orig, (0, 30))
        comparison_image.paste(img_repl, (w_img, 30))
        draw = ImageDraw.Draw(comparison_image)
        draw.text((10, 10), "Original Model", fill=(0, 0, 0))
        draw.text((w_img + 10, 10), f"{kind.upper()} Model", fill=(0, 0, 0))

    avg_mse = mse_accum / valid_count if valid_count else 0.0
    avg_cos = cos_accum / valid_count if valid_count else 0.0
    return avg_mse, avg_cos, comparison_image
