# `transcoder_training`

Training and loading of the timestep-conditioned transcoders (and the SAE baseline). Driven from the
repository-root `train_transcoder.py` / `train_sae.py`.

| File | What it holds |
|---|---|
| `transcoder.py` | `TemporalAwareTranscoder` — FiLM timestep modulation → ReLU encoder → unit-norm decoder; the `TemporalAwareSAE` baseline alias (same architecture, output-autoencoding objective); and `load_transcoders`. |
| `activation_store.py` | Dual-stream MLP activation capture (hooks on `ff` / `ff_context`) and the pinned harvesting buffer, with the per-step timestep context. |
| `train.py` | `TrainConfig` and the harvest → AdamW → CosineAnnealingLR loop; loss = variance-normalized MSE + L1, decoder renormalized each step. |
| `data.py` | Streaming of the prompt corpus and the fixed validation batch. |
| `evaluation.py` | Reconstruction / latent validation metrics. |
