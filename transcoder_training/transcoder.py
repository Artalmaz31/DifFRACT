import os
from typing import Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from diffusers.models.embeddings import get_timestep_embedding


class TemporalAwareTranscoder(nn.Module):
    """Wide, sparsely-activating, timestep-conditioned approximation of an MLP sublayer."""

    def __init__(self, d_model, expansion_factor, time_embed_dim, l1_coeff=None):
        super().__init__()
        self.d_model = d_model
        self.d_feat = int(d_model * expansion_factor)
        self.l1_coeff = l1_coeff
        self.time_embed_dim = time_embed_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )
        self.mod_linear = nn.Linear(time_embed_dim, d_model * 2)

        self.encoder = nn.Linear(d_model, self.d_feat, bias=True)
        self.decoder = nn.Linear(self.d_feat, d_model, bias=True)

        self._init_weights()
        self.normalize_decoder()

    def _init_weights(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.decoder.weight)
            self.encoder.weight.data = self.decoder.weight.data.t().clone()
            self.encoder.bias.data.zero_()
            self.decoder.bias.data.zero_()

            for m in self.time_mlp:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                    nn.init.zeros_(m.bias)

            nn.init.zeros_(self.mod_linear.weight)
            nn.init.zeros_(self.mod_linear.bias)

    def forward(self, x, t):
        t = t.to(dtype=torch.float32, device=x.device).view(-1)
        t_embed = get_timestep_embedding(t, embedding_dim=self.time_embed_dim).to(
            dtype=x.dtype, device=x.device
        )
        t_feat = self.time_mlp(t_embed)

        mod = self.mod_linear(t_feat)
        scale, shift = mod.chunk(2, dim=-1)
        x_mod = x * (1.0 + scale) + shift

        z = F.relu(self.encoder(x_mod))
        rec = self.decoder(z)
        return rec, z

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def get_modulation(self, t_emb: Tensor) -> Tuple[Tensor, Tensor]:
        t_feat = self.time_mlp(t_emb)
        mod = self.mod_linear(t_feat)
        scale, shift = mod.chunk(2, dim=-1)
        return scale, shift

    def encode(self, x: Tensor, t_emb: Tensor) -> Tuple[Tensor, Tensor]:
        scale, shift = self.get_modulation(t_emb)
        x_mod = x * (1.0 + scale) + shift
        h_pre = F.linear(x_mod, self.encoder.weight, self.encoder.bias)
        z = F.relu(h_pre)
        return z, h_pre

    def decode(self, z: Tensor) -> Tensor:
        return F.linear(z, self.decoder.weight, self.decoder.bias)

    def feature_preactivation(self, x: Tensor, t: Tensor, feat_idx: int) -> Tensor:
        """Pre-ReLU activation of a single feature across positions.

        Given an FF-input tensor x of shape (B, S, d_model) and the
        denoising timestep t, returns the modulated encoder pre-activation
        (x_mod @ w_enc) + b_enc of feature feat_idx, shape (B, S).
        Gradients flow through x (used by the VJP attribution path), so this
        is intentionally not wrapped in no_grad.
        """
        t = t.to(dtype=torch.float32, device=x.device).view(-1)
        t_emb = get_timestep_embedding(t, embedding_dim=self.time_embed_dim).to(
            dtype=x.dtype, device=x.device
        )
        scale, shift = self.get_modulation(t_emb)
        x_mod = x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        w_enc = self.encoder.weight[feat_idx].to(dtype=x.dtype)
        b_enc = self.encoder.bias[feat_idx].to(dtype=x.dtype)
        return (x_mod * w_enc).sum(dim=-1) + b_enc

    def forward_with_preact(
        self, x: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        t = t.to(dtype=torch.float32, device=x.device).view(-1)
        t_emb = get_timestep_embedding(t, embedding_dim=self.time_embed_dim).to(
            dtype=x.dtype, device=x.device
        )
        z, h_pre = self.encode(x, t_emb)
        rec = self.decode(z)
        return rec, z, h_pre

    def get_effective_encoder_bias(
        self, feat_idx: int, t: Tensor, block_shift: Optional[Tensor] = None
    ) -> float:
        device = self.encoder.weight.device
        tc_dtype = self.encoder.weight.dtype

        t = t.to(dtype=torch.float32, device=device).view(-1)
        t_emb = get_timestep_embedding(t, self.time_embed_dim).to(
            dtype=tc_dtype, device=device
        )
        scale_tc, shift_tc = self.get_modulation(t_emb)

        w_enc_i = self.encoder.weight[feat_idx]
        b_enc_i = self.encoder.bias[feat_idx]

        bias = b_enc_i + (w_enc_i * shift_tc[0]).sum()

        if block_shift is not None:
            block_shift = block_shift.to(device=device, dtype=tc_dtype)
            if block_shift.dim() >= 2:
                block_shift = block_shift[0]
            bias = bias + (w_enc_i * block_shift * (1.0 + scale_tc[0])).sum()

        return bias.item()

    @torch.no_grad()
    def _scale_shift(self, t, device, dtype):
        t_embed = get_timestep_embedding(t, embedding_dim=self.time_embed_dim).to(
            device=device
        )
        t_embed = t_embed.to(dtype=self.mod_linear.weight.dtype)
        t_feat = self.time_mlp(t_embed)
        mod = self.mod_linear(t_feat)
        scale, shift = mod.chunk(2, dim=-1)
        return scale.to(dtype=dtype), shift.to(dtype=dtype)

    @torch.no_grad()
    def _modulate(self, x, t):
        scale, shift = self._scale_shift(t, device=x.device, dtype=x.dtype)
        return x * (1.0 + scale[:, None, :]) + shift[:, None, :]

    @torch.no_grad()
    def encode_batch(self, x, t, feat_idx):
        x = x.to(dtype=self.encoder.weight.dtype)
        x_mod = self._modulate(x, t)
        W = self.encoder.weight.index_select(0, feat_idx)
        b = self.encoder.bias.index_select(0, feat_idx)
        return F.relu(F.linear(x_mod, W, b))

    @torch.no_grad()
    def encode_max(self, x, t, batch=2048):
        x = x.to(dtype=self.encoder.weight.dtype)
        x_mod = self._modulate(x, t)
        B, S, D = x_mod.shape
        W, b = self.encoder.weight, self.encoder.bias
        Fdim = W.shape[0]
        out = torch.empty((B, Fdim), device=x.device, dtype=torch.float32)
        for i in range(0, Fdim, batch):
            z = F.relu(F.linear(x_mod, W[i : i + batch], b[i : i + batch]))
            out[:, i : i + batch] = z.amax(dim=1).float()
            del z
        return out


def load_transcoders(
    transcoder_dir: str,
    layers: Sequence[int],
    *,
    d_model: int = 3072,
    expansion_factor: int = 16,
    time_embed_dim: int = 256,
    streams: Sequence[str] = ("img", "txt"),
    device="cpu",
    dtype=torch.float32,
    requires_grad: bool = False,
    skip_missing: bool = False,
) -> dict:
    """Load trained transcoders into a {f"{stream}_{layer}": module} dict."""
    transcoders = {}
    for layer in layers:
        for stream in streams:
            key = f"{stream}_{layer}"
            path = os.path.join(transcoder_dir, f"transcoder_{key}.pt")
            if skip_missing and not os.path.exists(path):
                continue
            tc = TemporalAwareTranscoder(
                d_model=d_model,
                expansion_factor=expansion_factor,
                time_embed_dim=time_embed_dim,
            )
            tc.load_state_dict(torch.load(path, map_location="cpu"))
            tc.to(device=device, dtype=dtype).eval()
            for p in tc.parameters():
                p.requires_grad_(requires_grad)
            transcoders[key] = tc
    return transcoders


# The SAE baseline used in the transcoder vs SAE comparison is architecturally identical to the transcoder
# The only difference is the training objective -- the SAE autoencodes the MLP output while the transcoder maps
# the MLP input to its output. That distinction lives entirely in the training loop so we expose the SAE as an alias
TemporalAwareSAE = TemporalAwareTranscoder
