from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from diffusers import FluxPipeline
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from .attribution_graph import NodeType, AggNodeId, AggAttributionGraph
from .replacement_model import FluxTrace, LRMConfig, LRMPatcher
from .influence import (
    build_node_index,
    build_adjacency,
    normalized_adjacency,
    indirect_influence,
)


def decode_latents(
    pipe: FluxPipeline, latents: torch.Tensor, height: int = 512, width: int = 512
) -> np.ndarray:
    with torch.no_grad():
        lat = latents.to(device=pipe.vae.device, dtype=pipe.vae.dtype)

        batch_size = lat.shape[0]
        h_latent = height // 8
        w_latent = width // 8
        channels = lat.shape[-1]

        lat = lat.view(batch_size, h_latent // 2, w_latent // 2, channels // 4, 2, 2)
        lat = lat.permute(0, 3, 1, 4, 2, 5)
        lat = lat.reshape(batch_size, channels // 4, h_latent, w_latent)

        lat = lat / pipe.vae.config.scaling_factor

        image = pipe.vae.decode(lat, return_dict=False)[0]

        image = (image.float().cpu() / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype(np.uint8)

        return image[0]


class StepAwareLRMToggle:
    def __init__(self, pipe: FluxPipeline, patcher: "LRMPatcher", target_step: int):
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.patcher = patcher
        self.target_step = int(target_step)

        self._call_idx = -1
        self._active = False
        self._h_pre = None
        self._h_post = None

    def install(self):
        def pre_hook(module, args, kwargs):
            self._call_idx += 1
            if self._call_idx == self.target_step and not self._active:
                self._active = True
                self.patcher.__enter__()
            return None

        def post_hook(module, args, kwargs, output):
            if self._active and self._call_idx == self.target_step:
                self.patcher.__exit__(None, None, None)
                self._active = False
            return None

        self._h_pre = self.transformer.register_forward_pre_hook(
            pre_hook, with_kwargs=True
        )
        self._h_post = self.transformer.register_forward_hook(
            post_hook, with_kwargs=True
        )

    def remove(self):
        if self._h_pre is not None:
            self._h_pre.remove()
            self._h_pre = None
        if self._h_post is not None:
            self._h_post.remove()
            self._h_post = None


@torch.no_grad()
def generate_comparison_images(
    pipe: FluxPipeline,
    patcher: "LRMPatcher",
    cfg: LRMConfig,
    prompt: str,
    seed: int,
    target_step: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    def run_pipeline(use_lrm: bool) -> torch.Tensor:
        gen = torch.Generator(device=cfg.device).manual_seed(seed)
        toggle = None
        if use_lrm:
            toggle = StepAwareLRMToggle(pipe, patcher, target_step)
            toggle.install()

        result = pipe(
            prompt,
            prompt_2=prompt,
            height=cfg.height,
            width=cfg.width,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            output_type="latent",
            generator=gen,
        )
        return result.images.detach().float().cpu()

    print("  Generating original image...")
    lat_orig = run_pipeline(use_lrm=False)

    print("  Generating LRM image...")
    lat_lrm = run_pipeline(use_lrm=True)

    lat_diff = (lat_lrm - lat_orig).abs()
    stats = {
        "latent_max_diff": float(lat_diff.max().item()),
        "latent_mean_diff": float(lat_diff.mean().item()),
        "latent_cos_sim": float(
            F.cosine_similarity(lat_orig.reshape(1, -1), lat_lrm.reshape(1, -1)).item()
        ),
    }

    print("  Decoding images...")
    img_orig = decode_latents(pipe, lat_orig.to(cfg.device), cfg.height, cfg.width)
    img_lrm = decode_latents(pipe, lat_lrm.to(cfg.device), cfg.height, cfg.width)

    return img_orig, img_lrm, stats


def plot_comparison(
    img_orig: np.ndarray,
    img_lrm: np.ndarray,
    stats: Dict[str, float],
    prompt: str,
    target_step: int,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_orig)
    axes[0].set_title("Original Model", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(img_lrm)
    axes[1].set_title(f"LRM @ step={target_step}", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    diff = np.abs(img_orig.astype(np.float32) - img_lrm.astype(np.float32))
    diff_gray = diff.mean(axis=-1)

    im = axes[2].imshow(
        diff_gray, cmap="hot", vmin=0, vmax=max(float(diff_gray.max()), 1.0)
    )
    axes[2].set_title("Pixel Difference", fontsize=14, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    pixel_max = float(diff.max())
    pixel_mean = float(diff.mean())
    mse = float((diff**2).mean())
    psnr = float(10 * np.log10(255**2 / max(mse, 1e-10)))

    plt.show()
    plt.close(fig)

    return {
        "pixel_psnr_db": psnr,
        "pixel_max_diff": pixel_max,
        "pixel_mean_diff": pixel_mean,
        **stats,
    }


class LRMValidator:
    def __init__(self, cfg: LRMConfig):
        self.cfg = cfg

    def validate_attention_accuracy(
        self,
        trace: FluxTrace,
        *,
        layers: Optional[List[int]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        if layers is None:
            layers = list(sorted(trace.layer_caches.keys()))

        per_layer = {}
        for layer_idx in layers:
            lc = trace.get_layer(layer_idx)
            ac = lc.attention
            if ac is None:
                continue

            e_img = ac.attn_error_img.float()
            e_txt = ac.attn_error_txt.float()

            img_max = float(e_img.abs().max().item())
            txt_max = float(e_txt.abs().max().item())
            img_mean = float(e_img.abs().mean().item())
            txt_mean = float(e_txt.abs().mean().item())

            per_layer[layer_idx] = {
                "attn_error_img_max": img_max,
                "attn_error_img_mean": img_mean,
                "attn_error_txt_max": txt_max,
                "attn_error_txt_mean": txt_mean,
            }

        if verbose:
            print("Attention accuracy (frozen vs original)")
            for k in sorted(per_layer.keys()):
                r = per_layer[k]
                print(
                    f"  L{k}: "
                    f"img_max={r['attn_error_img_max']:.3g} img_mean={r['attn_error_img_mean']:.3g} | "
                    f"txt_max={r['attn_error_txt_max']:.3g} txt_mean={r['attn_error_txt_mean']:.3g} | "
                )

        return per_layer

    def validate_lrm_exact(
        self,
        transformer: nn.Module,
        trace: FluxTrace,
        transcoders: Dict[str, nn.Module],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        kwargs = dict(trace.transformer_kwargs)

        original_outputs = {}

        def capture_original_hook(layer_idx):
            def hook(module, args, kwargs, output):
                original_outputs[layer_idx] = {
                    "img": (
                        output[0].detach().clone()
                        if isinstance(output, tuple)
                        else output.detach().clone()
                    ),
                    "txt": (
                        output[1].detach().clone()
                        if isinstance(output, tuple) and len(output) > 1
                        else None
                    ),
                }

            return hook

        hooks = []
        for layer_idx in self.cfg.target_layers:
            blk = transformer.transformer_blocks[layer_idx]
            h = blk.register_forward_hook(
                capture_original_hook(layer_idx), with_kwargs=True
            )
            hooks.append(h)

        with torch.no_grad():
            original_out = transformer(**kwargs)

        for h in hooks:
            h.remove()

        lrm_outputs = {}

        def capture_lrm_hook(layer_idx):
            def hook(module, args, kwargs, output):
                lrm_outputs[layer_idx] = {
                    "img": (
                        output[0].detach().clone()
                        if isinstance(output, tuple)
                        else output.detach().clone()
                    ),
                    "txt": (
                        output[1].detach().clone()
                        if isinstance(output, tuple) and len(output) > 1
                        else None
                    ),
                }

            return hook

        with LRMPatcher(transformer, trace, transcoders, self.cfg, mode="exact"):
            hooks = []
            for layer_idx in self.cfg.target_layers:
                blk = transformer.transformer_blocks[layer_idx]
                h = blk.register_forward_hook(
                    capture_lrm_hook(layer_idx), with_kwargs=True
                )
                hooks.append(h)

            with torch.no_grad():
                lrm_out = transformer(**kwargs)

            for h in hooks:
                h.remove()

        results = {
            "transformer_output": {
                "max_err": (lrm_out[0] - original_out[0]).abs().max().item(),
                "mean_err": (lrm_out[0] - original_out[0]).abs().mean().item(),
            },
            "per_layer": {},
        }

        for layer_idx in self.cfg.target_layers:
            if layer_idx in original_outputs and layer_idx in lrm_outputs:
                orig = original_outputs[layer_idx]
                lrm = lrm_outputs[layer_idx]

                layer_result = {}
                if orig["img"] is not None and lrm["img"] is not None:
                    layer_result["img_max_err"] = (
                        (lrm["img"] - orig["img"]).abs().max().item()
                    )
                    layer_result["img_mean_err"] = (
                        (lrm["img"] - orig["img"]).abs().mean().item()
                    )
                if orig["txt"] is not None and lrm["txt"] is not None:
                    layer_result["txt_max_err"] = (
                        (lrm["txt"] - orig["txt"]).abs().max().item()
                    )
                    layer_result["txt_mean_err"] = (
                        (lrm["txt"] - orig["txt"]).abs().mean().item()
                    )

                results["per_layer"][layer_idx] = layer_result

        if verbose:
            print("Local Replacement Model accuracy")
            print(
                f"Transformer output: "
                f"max_err={results['transformer_output']['max_err']:.3g}, "
                f"mean_err={results['transformer_output']['mean_err']:.3g}"
            )
            for layer_idx, r in results["per_layer"].items():
                print(
                    f"  L{layer_idx}: "
                    f"img_max={r['img_max_err']:.3g} img_mean={r['img_mean_err']:.3g} | "
                    f"txt_max={r['txt_max_err']:.3g} txt_mean={r['txt_mean_err']:.3g} | "
                )

        return results

    def validate_attribution_sum(
        self,
        graph,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Validate that edges to the ultimate target sum to h_pre - bias."""
        expected = graph.target_preactivation - graph.target_encoder_bias
        target_edges = [e for e in graph.edges if e.target_id == graph.target_id]
        actual = sum(e.attribution for e in target_edges)

        abs_err = abs(actual - expected)
        rel_err = abs_err / (abs(expected) + 1e-8)

        results = {
            "expected": expected,
            "actual": actual,
            "abs_error": abs_err,
            "rel_error": rel_err,
            "n_target_edges": len(target_edges),
            "n_total_edges": len(graph.edges),
        }

        if verbose:
            print(
                f"Attribution sum (edges to target): "
                f"expected={expected:.4f}, actual={actual:.4f}, "
                f"rel_err={rel_err:.2%} "
                f"({len(target_edges)}/{len(graph.edges)} edges)"
            )

        return results

    def validate_images_orig_vs_lrm(
        self,
        pipe: FluxPipeline,
        trace: FluxTrace,
        transcoders: Dict[str, nn.Module],
        *,
        prompt: Optional[str] = None,
        seed: Optional[int] = None,
        target_step: Optional[int] = None,
        lrm_mode: str = "linear",
        save_path: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        if prompt is None:
            prompt = trace.prompt
        if seed is None:
            seed = trace.seed
        if target_step is None:
            target_step = trace.step_idx

        patcher = LRMPatcher(
            pipe.transformer, trace, transcoders, self.cfg, mode=lrm_mode
        )

        if verbose:
            print("Image comparison: original vs LRM")
            print(f"  prompt={prompt!r}")
            print(f"  seed={seed} target_step={target_step}")

        img_orig, img_lrm, lat_stats = generate_comparison_images(
            pipe=pipe,
            patcher=patcher,
            cfg=self.cfg,
            prompt=prompt,
            seed=seed,
            target_step=target_step,
        )

        pixel_stats = plot_comparison(
            img_orig=img_orig,
            img_lrm=img_lrm,
            stats=lat_stats,
            prompt=prompt,
            target_step=target_step,
            save_path=save_path,
        )
        return pixel_stats


class PerturbationValidator:
    """Validates attribution graphs through feature ablation in the original model."""

    def __init__(
        self,
        transformer: nn.Module,
        transcoders: Dict[str, nn.Module],
        cfg: LRMConfig,
    ):
        self.transformer = transformer
        self.transcoders = transcoders
        self.cfg = cfg

    @staticmethod
    def _get_best_position(
        node: "AggNodeId",
        graph: "AggAttributionGraph",
    ) -> int:
        """Return the position with highest |activation| for a node."""
        act_map = graph.activation_maps.get(str(node), {})
        if act_map:
            return max(act_map, key=lambda p: abs(act_map[p]))
        return 0

    def validate(
        self,
        trace: FluxTrace,
        graph: "AggAttributionGraph",
        top_k: int = 30,
        verbose: bool = True,
    ) -> Dict[str, Any]:

        # Collect all feature nodes in the graph
        feature_nodes = []
        for key, info in graph.nodes.items():
            nid = info["id"]
            if nid.node_type == NodeType.FEATURE:
                feature_nodes.append(nid)

        # Limit to top_k most influential sources
        if len(feature_nodes) > top_k:
            inf = self._compute_pairwise_influence(graph, feature_nodes)
            node_importance = {}
            for i, src in enumerate(feature_nodes):
                node_importance[src] = sum(
                    abs(inf[i, j]) for j in range(len(feature_nodes)) if j != i
                )
            feature_nodes.sort(key=lambda n: node_importance.get(n, 0), reverse=True)
            feature_nodes = feature_nodes[:top_k]

        N = len(feature_nodes)
        if N < 2:
            if verbose:
                print("Not enough feature nodes for pairwise validation")
            return {"spearman_r": 0.0, "n_pairs": 0}

        if verbose:
            print(
                f"Pairwise validation: {N} feature nodes, "
                f"{N * (N-1)} pairs, {N+1} forward passes"
            )

        # Compute graph-based pairwise indirect influence
        influence_matrix = self._compute_pairwise_influence(graph, feature_nodes)

        # Get positions
        if feature_nodes and hasattr(feature_nodes[0], "position"):
            node_positions = {nid: nid.position for nid in feature_nodes}
        else:
            node_positions = {
                nid: self._get_best_position(nid, graph) for nid in feature_nodes
            }

        # Measure baseline h_pre for all features
        baseline = self._measure_all_features(
            trace, feature_nodes, node_positions, ablation=None
        )

        # Ablate each source and measure effect on all targets
        actual_effect = np.zeros((N, N), dtype=np.float64)
        for i, src in enumerate(feature_nodes):
            pos = node_positions[src]
            ablation = (src.layer, src.stream, pos, src.feat_idx)
            ablated = self._measure_all_features(
                trace, feature_nodes, node_positions, ablation=ablation
            )

            for j, tgt in enumerate(feature_nodes):
                if i != j:
                    actual_effect[i, j] = abs(ablated[j] - baseline[j])

            if verbose and (i + 1) % 10 == 0:
                print(f"  {i + 1}/{N} ablations done")

        # Collect all (predicted, actual) pairs
        predicted_list = []
        actual_list = []
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                predicted_list.append(influence_matrix[i, j])
                actual_list.append(actual_effect[i, j])

        predicted = np.array(predicted_list)
        actual = np.array(actual_list)

        # Filter out zero pairs
        mask = (np.abs(predicted) > 1e-12) | (np.abs(actual) > 1e-12)
        if mask.sum() < 3:
            if verbose:
                print("Not enough non-zero pairs for correlation")
            return {"spearman_r": 0.0, "n_pairs": int(mask.sum())}

        sp = scipy_stats.spearmanr(predicted[mask], actual[mask])
        pr = scipy_stats.pearsonr(predicted[mask], actual[mask])

        summary = {
            "spearman_r": float(sp.correlation),
            "spearman_p": float(sp.pvalue),
            "pearson_r": float(pr.statistic),
            "pearson_p": float(pr.pvalue),
            "n_features": N,
            "n_pairs": int(mask.sum()),
            "n_forward_passes": N + 1,
            "feature_nodes": [str(n) for n in feature_nodes],
        }

        if verbose:
            print("\nPairwise validation results:")
            print(f"  Spearman r = {sp.correlation:.4f}  (p = {sp.pvalue:.2e})")
            print(f"  Pearson  r = {pr.statistic:.4f}  (p = {pr.pvalue:.2e})")
            print(f"  {N} features, {int(mask.sum())} non-zero pairs")

        return summary

    def validate_direct(
        self,
        trace: FluxTrace,
        graph: "AggAttributionGraph",
        top_k: int = 30,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        target = graph.target_id
        target_pos = graph.target_position

        baseline_h_pre = self._measure_target_h_pre(
            trace,
            target.layer,
            target.stream,
            target.feat_idx,
            target_pos,
            ablation=None,
        )

        target_feature_edges = [
            e
            for e in graph.edges
            if e.target_id == target and e.source_id.node_type == NodeType.FEATURE
        ]
        target_feature_edges.sort(key=lambda e: abs(e.attribution), reverse=True)
        top_edges = target_feature_edges[:top_k]

        if verbose:
            print(
                f"Direct validation: {len(top_edges)} features "
                f"(baseline h_pre = {baseline_h_pre:.4f})"
            )

        results: List[Dict[str, Any]] = []
        for i, edge in enumerate(top_edges):
            src = edge.source_id
            pos = self._get_best_position(src, graph)
            ablation = (src.layer, src.stream, pos, src.feat_idx)

            ablated_h_pre = self._measure_target_h_pre(
                trace,
                target.layer,
                target.stream,
                target.feat_idx,
                target_pos,
                ablation=ablation,
            )

            predicted_delta = -edge.attribution
            actual_delta = ablated_h_pre - baseline_h_pre

            results.append(
                {
                    "source": str(src),
                    "attribution": edge.attribution,
                    "predicted_delta": predicted_delta,
                    "actual_delta": actual_delta,
                }
            )

            if verbose and (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(top_edges)} ablations done")

        predicted = np.array([r["predicted_delta"] for r in results])
        actual = np.array([r["actual_delta"] for r in results])

        sp = scipy_stats.spearmanr(predicted, actual)
        pr = scipy_stats.pearsonr(predicted, actual)

        summary = {
            "spearman_r": float(sp.correlation),
            "spearman_p": float(sp.pvalue),
            "pearson_r": float(pr.statistic),
            "pearson_p": float(pr.pvalue),
            "baseline_h_pre": float(baseline_h_pre),
            "n_ablations": len(results),
            "results": results,
        }

        if verbose:
            print("\nDirect validation results:")
            print(f"  Spearman r = {sp.correlation:.4f}  (p = {sp.pvalue:.2e})")
            print(f"  Pearson  r = {pr.statistic:.4f}  (p = {pr.pvalue:.2e})")

        return summary

    def plot_results(
        self,
        summary: Dict[str, Any],
        title: str = "Perturbation Validation",
    ):
        if "results" in summary:
            results = summary["results"]
            predicted = np.array([r["predicted_delta"] for r in results])
            actual = np.array([r["actual_delta"] for r in results])
            xlabel = "Predicted delta h_pre  (-attribution)"
            ylabel = "Actual delta h_pre  (ablated - baseline)"
        else:
            print(f"Pairwise validation: Spearman r = {summary['spearman_r']:.4f}")
            return

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.scatter(predicted, actual, alpha=0.6, s=30, edgecolors="k", linewidths=0.3)

        lims = [
            min(predicted.min(), actual.min()) * 1.1,
            max(predicted.max(), actual.max()) * 1.1,
        ]
        ax.plot(lims, lims, "r--", linewidth=1, label="y = x (perfect)")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(
            f"{title}\n"
            f"Spearman r = {summary['spearman_r']:.3f}, "
            f"Pearson r = {summary['pearson_r']:.3f}, "
            f"n = {summary.get('n_ablations', summary.get('n_pairs', 0))}",
            fontsize=11,
        )
        ax.legend(fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    @torch.no_grad()
    def _measure_target_h_pre(
        self,
        trace: FluxTrace,
        target_layer: int,
        target_stream: str,
        target_feat_idx: int,
        target_position: int,
        ablation: Optional[Tuple[int, str, int, int]] = None,
    ) -> float:
        device = self.cfg.device
        x_ff_box: Dict[str, Optional[Tensor]] = {"value": None}
        hooks = []
        tcs_on_gpu: List[nn.Module] = []

        try:
            if ablation is not None:
                abl_tc = self.transcoders[f"{ablation[1]}_{ablation[0]}"]
                abl_tc.to(device)
                tcs_on_gpu.append(abl_tc)
                self._install_ablation_hooks(trace, ablation, hooks)

            target_blk = self.transformer.transformer_blocks[target_layer]
            target_ff = (
                target_blk.ff if target_stream == "img" else target_blk.ff_context
            )

            def target_ff_pre_hook(module, args):
                x_ff_box["value"] = args[0]

            hooks.append(target_ff.register_forward_pre_hook(target_ff_pre_hook))

            kwargs = dict(trace.transformer_kwargs)
            self.transformer(**kwargs)

        finally:
            for h in hooks:
                h.remove()
            for tc in tcs_on_gpu:
                tc.cpu()

        x_ff = x_ff_box["value"]
        if x_ff is None:
            raise RuntimeError(
                f"Failed to capture x_ff at L{target_layer}/{target_stream}"
            )

        return self._compute_h_pre_from_x_ff(
            trace,
            x_ff,
            target_layer,
            target_stream,
            target_feat_idx,
            target_position,
        )

    @torch.no_grad()
    def _measure_all_features(
        self,
        trace: FluxTrace,
        feature_nodes: List["AggNodeId"],
        node_positions: Dict["AggNodeId", int],
        ablation: Optional[Tuple[int, str, int, int]] = None,
    ) -> List[float]:
        device = self.cfg.device
        hooks = []
        tcs_on_gpu: List[nn.Module] = []
        layer_stream_feats: Dict[Tuple[int, str], List[int]] = {}
        for i, node in enumerate(feature_nodes):
            key = (node.layer, node.stream)
            if key not in layer_stream_feats:
                layer_stream_feats[key] = []
            layer_stream_feats[key].append(i)

        captured_x_ff: Dict[Tuple[int, str], Tensor] = {}

        try:
            if ablation is not None:
                abl_tc = self.transcoders[f"{ablation[1]}_{ablation[0]}"]
                abl_tc.to(device)
                tcs_on_gpu.append(abl_tc)
                self._install_ablation_hooks(trace, ablation, hooks)

            for layer, stream in layer_stream_feats.keys():
                blk = self.transformer.transformer_blocks[layer]
                ff = blk.ff if stream == "img" else blk.ff_context

                def make_hook(l, s):
                    def hook_fn(module, args):
                        captured_x_ff[(l, s)] = args[0]

                    return hook_fn

                hooks.append(ff.register_forward_pre_hook(make_hook(layer, stream)))

            kwargs = dict(trace.transformer_kwargs)
            self.transformer(**kwargs)

        finally:
            for h in hooks:
                h.remove()
            for tc in tcs_on_gpu:
                tc.cpu()

        results = [0.0] * len(feature_nodes)
        for (layer, stream), indices in layer_stream_feats.items():
            x_ff = captured_x_ff.get((layer, stream))
            if x_ff is None:
                continue
            tc = self.transcoders[f"{stream}_{layer}"]
            tc.to(device)
            for idx in indices:
                node = feature_nodes[idx]
                pos = node_positions[node]
                results[idx] = self._compute_h_pre_from_x_ff(
                    trace,
                    x_ff,
                    node.layer,
                    node.stream,
                    node.feat_idx,
                    pos,
                )
            tc.cpu()

        return results

    def _install_ablation_hooks(
        self,
        trace: FluxTrace,
        ablation: Tuple[int, str, int, int],
        hooks: list,
    ) -> None:
        abl_layer, abl_stream, abl_pos, abl_feat = ablation
        abl_blk = self.transformer.transformer_blocks[abl_layer]
        abl_ff = abl_blk.ff if abl_stream == "img" else abl_blk.ff_context
        tc = self.transcoders[f"{abl_stream}_{abl_layer}"]
        device = self.cfg.device

        captured_input: Dict[str, Optional[Tensor]] = {"x": None}

        def ff_pre_hook(module, args):
            captured_input["x"] = args[0]

        hooks.append(abl_ff.register_forward_pre_hook(ff_pre_hook))

        def ff_post_hook(module, args, output):
            x_ff = captured_input["x"]
            if x_ff is None:
                return output

            tc_dtype = next(tc.parameters()).dtype
            x_tc = x_ff.to(dtype=tc_dtype, device=device)

            t = trace.timestep_tensor.to(device=device, dtype=torch.float32).view(-1)
            h_pre = tc.feature_preactivation(x_tc, t, abl_feat)[0, abl_pos]
            z_f = F.relu(h_pre)

            if z_f.abs().item() < 1e-10:
                return output

            W_dec_f = tc.decoder.weight[:, abl_feat].to(
                dtype=output.dtype, device=device
            )
            delta = z_f.to(dtype=output.dtype) * W_dec_f

            modified = output.clone()
            modified[0, abl_pos] -= delta
            return modified

        hooks.append(abl_ff.register_forward_hook(ff_post_hook))

    def _compute_h_pre_from_x_ff(
        self,
        trace: FluxTrace,
        x_ff: Tensor,
        target_layer: int,
        target_stream: str,
        target_feat_idx: int,
        target_position: int,
    ) -> float:
        """Compute a feature's preactivation from the FF input tensor."""
        device = self.cfg.device
        tc = self.transcoders[f"{target_stream}_{target_layer}"]
        tc_was_on_cpu = next(tc.parameters()).device.type == "cpu"
        if tc_was_on_cpu:
            tc.to(device)
        try:
            tc_dtype = next(tc.parameters()).dtype
            x_ff_tc = x_ff.to(dtype=tc_dtype, device=device)

            t = trace.timestep_tensor.to(device=device, dtype=torch.float32).view(-1)
            h_pre = tc.feature_preactivation(x_ff_tc, t, target_feat_idx)[
                0, target_position
            ]
            return float(h_pre.item())
        finally:
            if tc_was_on_cpu:
                tc.cpu()

    def _compute_pairwise_influence(
        self,
        graph: "AggAttributionGraph",
        feature_nodes: List["AggNodeId"],
    ) -> np.ndarray:
        _, node_to_idx = build_node_index(graph)
        A_norm = normalized_adjacency(build_adjacency(graph.edges, node_to_idx))
        B = indirect_influence(A_norm)

        N = len(feature_nodes)
        result = np.zeros((N, N), dtype=np.float64)
        for i, src in enumerate(feature_nodes):
            si = node_to_idx.get(src)
            if si is None:
                continue
            for j, tgt in enumerate(feature_nodes):
                ti = node_to_idx.get(tgt)
                if ti is None:
                    continue
                result[i, j] = B[si, ti]

        return result
