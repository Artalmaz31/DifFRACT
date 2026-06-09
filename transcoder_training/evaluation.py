import gc
from typing import Dict, List, Tuple
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
    lat = _unpack_flux_latents(lat, height, width)
    lat = lat / pipe.vae.config.scaling_factor
    with torch.inference_mode():
        image = pipe.vae.decode(lat).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return Image.fromarray((image[0] * 255).round().astype("uint8"))


@torch.no_grad()
def run_validation(
    pipe,
    models: Dict[str, nn.Module],
    target_layers,
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
) -> Tuple[float, float, "Image.Image"]:
    """Replace all target MLPs with models and compare final latents to the original."""
    torch.cuda.empty_cache()
    gc.collect()
    models.eval()

    prev_enabled = capturer.enabled
    capturer.enabled = False

    backup_layers: Dict[str, nn.Module] = {}
    repl_layers: Dict[str, nn.Module] = {}
    for l in target_layers:
        block = pipe.transformer.transformer_blocks[l]
        backup_layers[f"img_{l}"] = block.ff
        backup_layers[f"txt_{l}"] = block.ff_context
        for stream, orig in (("img", block.ff), ("txt", block.ff_context)):
            model = models[f"{stream}_{l}"]
            if kind == "sae":
                repl_layers[f"{stream}_{l}"] = SAEInferenceWrapper(
                    model, orig, orig_dtype, t_ctx
                )
            else:
                repl_layers[f"{stream}_{l}"] = TranscoderInferenceWrapper(
                    model, orig_dtype, t_ctx
                )

    def set_model_to_original():
        for l in target_layers:
            block = pipe.transformer.transformer_blocks[l]
            block.ff = backup_layers[f"img_{l}"]
            block.ff_context = backup_layers[f"txt_{l}"]

    def set_model_to_replacement():
        for l in target_layers:
            block = pipe.transformer.transformer_blocks[l]
            block.ff = repl_layers[f"img_{l}"]
            block.ff_context = repl_layers[f"txt_{l}"]

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

        set_model_to_original()
        with torch.inference_mode():
            lat_orig = pipe(
                batch_prompts,
                prompt_2=batch_prompts,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=0.0,
                output_type="latent",
                generator=_gen(),
            ).images.cpu()
        if i == 0:
            viz_orig = lat_orig[0].clone()

        set_model_to_replacement()
        with torch.inference_mode():
            lat_repl = pipe(
                batch_prompts,
                prompt_2=batch_prompts,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=0.0,
                output_type="latent",
                generator=_gen(),
            ).images.cpu()
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
    repl_layers.clear()
    torch.cuda.empty_cache()
    gc.collect()

    comparison_image = None
    if viz_orig is not None and viz_repl is not None:
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

    capturer.enabled = prev_enabled
    return avg_mse, avg_cos, comparison_image
