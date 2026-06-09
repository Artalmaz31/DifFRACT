from typing import Dict, Sequence
import torch


class TimestepContext:
    """Holds the current denoising timestep, set by a transformer forward pre-hook."""

    def __init__(self):
        self.t = None
        self.t_cpu = None

    @torch.no_grad()
    def set(self, t):
        t = t.detach().to(dtype=torch.float32)
        self.t = t
        self.t_cpu = t.to("cpu", dtype=torch.float32)


def install_timestep_hook(pipe, t_ctx: TimestepContext):
    """Register a forward pre-hook that records each transformer call's timestep."""

    def _transformer_pre_hook(module, args, kwargs):
        t_ctx.set(kwargs["timestep"])

    return pipe.transformer.register_forward_pre_hook(
        _transformer_pre_hook, with_kwargs=True
    )


def make_buffers(
    target_layers: Sequence[int],
    d_model: int,
    buffer_size: int,
    buffer_dtype: torch.dtype,
) -> Dict[str, dict]:
    """Allocate per-(layer, stream) pinned replay buffers of (x, y, t) records."""
    buffers: Dict[str, dict] = {}
    for l in target_layers:
        for stream in ("img", "txt"):
            buffers[f"{stream}_{l}"] = {
                "x": torch.empty(
                    (buffer_size, d_model), dtype=buffer_dtype
                ).pin_memory(),
                "y": torch.empty(
                    (buffer_size, d_model), dtype=buffer_dtype
                ).pin_memory(),
                "t": torch.empty((buffer_size,), dtype=torch.float32).pin_memory(),
                "ptr": 0,
            }
    return buffers


class DualStreamCapture:
    """Forward hooks capturing (x, MLP(x), t) for both streams of each target block."""

    def __init__(self, pipe, layers, buffers, buffer_dtype, t_ctx: TimestepContext):
        self.hooks = []
        self.enabled = True
        self.buffers = buffers
        self.buffer_dtype = buffer_dtype
        self.t_ctx = t_ctx

        for l in layers:
            blk = pipe.transformer.transformer_blocks[l]
            self.hooks.append(blk.ff.register_forward_hook(self.make_hook(f"img_{l}")))
            self.hooks.append(
                blk.ff_context.register_forward_hook(self.make_hook(f"txt_{l}"))
            )

    def make_hook(self, key):
        def hook_fn(module, args, output):
            if not self.enabled:
                return

            t_cpu = self.t_ctx.t_cpu
            x = args[0]
            y = output

            B, S, D = x.shape

            x_flat = x.detach().reshape(B * S, D)
            y_flat = y.detach().reshape(B * S, D)
            t_flat = t_cpu.reshape(B).repeat_interleave(S)

            buf = self.buffers[key]
            cap = buf["x"].shape[0]
            ptr = int(buf["ptr"])
            if ptr >= cap:
                return

            take = min(x_flat.shape[0], cap - ptr)
            if take <= 0:
                return

            x_flat = x_flat.to(self.buffer_dtype)
            y_flat = y_flat.to(self.buffer_dtype)
            t_flat = t_flat.to(torch.float32)

            buf["x"][ptr : ptr + take].copy_(x_flat[:take], non_blocking=True)
            buf["y"][ptr : ptr + take].copy_(y_flat[:take], non_blocking=True)
            buf["t"][ptr : ptr + take].copy_(t_flat[:take], non_blocking=True)
            buf["ptr"] = ptr + take

        return hook_fn

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
