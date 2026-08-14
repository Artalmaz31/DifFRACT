from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ArchSpec:
    name: str
    pipeline_cls: str
    default_model_id: str
    guidance_scale: float
    num_inference_steps: int
    d_model: int
    timestep_scale: float = 1.0
    l1_coeff: Optional[Dict[str, float]] = None
    lr: Optional[Dict[str, float]] = None
    prompt_aliases: Tuple[str, ...] = ("prompt_2",)
    buffer_size: int = 1_000_000
    prompts_per_inference: int = 32


def runs_real_cfg(pipeline_cls: str, guidance_scale: float) -> bool:
    """True when the pipeline stacks [uncond, cond] and so doubles the transformer batch."""
    return pipeline_cls == "StableDiffusion3Pipeline" and guidance_scale > 1


FLUX_SCHNELL = ArchSpec(
    name="flux-schnell",
    pipeline_cls="FluxPipeline",
    default_model_id="black-forest-labs/FLUX.1-schnell",
    d_model=3072,
    guidance_scale=0.0,
    num_inference_steps=4,
    buffer_size=1_000_000,
    prompts_per_inference=32,
)

FLUX_DEV = ArchSpec(
    name="flux-dev",
    pipeline_cls="FluxPipeline",
    default_model_id="black-forest-labs/FLUX.1-dev",
    d_model=3072,
    guidance_scale=3.5,
    num_inference_steps=50,
    buffer_size=1_638_400,
    prompts_per_inference=32,
)

SD3_MEDIUM = ArchSpec(
    name="sd3-medium",
    pipeline_cls="StableDiffusion3Pipeline",
    default_model_id="stabilityai/stable-diffusion-3-medium-diffusers",
    d_model=1536,
    guidance_scale=7.0,
    num_inference_steps=28,
    timestep_scale=1000.0,
    prompt_aliases=("prompt_2", "prompt_3"),
    buffer_size=1_835_008,
    prompts_per_inference=32,
)

SD35_MEDIUM = ArchSpec(
    name="sd3.5-medium",
    pipeline_cls="StableDiffusion3Pipeline",
    default_model_id="stabilityai/stable-diffusion-3.5-medium",
    d_model=1536,
    guidance_scale=4.5,
    num_inference_steps=40,
    timestep_scale=1000.0,
    prompt_aliases=("prompt_2", "prompt_3"),
    buffer_size=2_621_440,
    prompts_per_inference=32,
)

ARCHS = {
    spec.name: spec for spec in (FLUX_SCHNELL, FLUX_DEV, SD3_MEDIUM, SD35_MEDIUM)
}
