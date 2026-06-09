import os
import gc
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import pandas as pd
from time import time
import torch
from diffusers import FluxPipeline
from transcoder_training.transcoder import TemporalAwareTranscoder, load_transcoders
from .attribution_graph import AttributionGraph, AggAttributionGraph, GraphAggregator
from .tracing import ExpansionConfig, CircuitTracer
from .pruning import GraphPruner
from .replacement_model import FluxTrace, LRMConfig, FluxTraceCapturer


def infer_position_for_feature(
    trace,
    layer: int,
    stream: str,
    feat_idx: int,
) -> Tuple[int, float]:
    lc = trace.get_layer(layer)
    h_pre = getattr(lc, f"{stream}_h_pre", None)
    acts = h_pre[0, :, feat_idx].float().cpu()

    pos = int(torch.argmax(acts).item())
    score = float(acts[pos].item())
    return pos, score


class FluxLRMPipeline:
    def __init__(self, cfg: LRMConfig):
        self.cfg = cfg

        self.pipe: Optional["FluxPipeline"] = None
        self.transcoders: Dict[str, TemporalAwareTranscoder] = {}
        self.capturer: Optional[FluxTraceCapturer] = None
        self.circuit_tracer: Optional[CircuitTracer] = None
        self.perturbation_validator: Optional[Any] = None
        self.pruner: Optional[GraphPruner] = None
        self.validator: Optional[Any] = None

    def initialize(self):
        # Tracing runs in float32 with TF32 disabled for attribution accuracy.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        self.pipe = FluxPipeline.from_pretrained(
            self.cfg.model_id,
            torch_dtype=self.cfg.dtype,
        ).to(self.cfg.device)

    def load_transcoders(self):
        self.transcoders = load_transcoders(
            self.cfg.transcoder_dir,
            self.cfg.target_layers,
            d_model=self.cfg.d_model,
            expansion_factor=self.cfg.expansion_factor,
            time_embed_dim=self.cfg.time_embed_dim,
            device="cpu",
            dtype=torch.float32,
            requires_grad=False,
        )

        from .validation import LRMValidator, PerturbationValidator

        self.capturer = FluxTraceCapturer(self.pipe, self.transcoders, self.cfg)
        self.circuit_tracer = CircuitTracer(
            self.pipe.transformer, self.transcoders, self.cfg
        )
        self.perturbation_validator = PerturbationValidator(
            self.pipe.transformer, self.transcoders, self.cfg
        )
        self.pruner = GraphPruner(
            node_threshold_img=self.cfg.prune_node_threshold_img,
            node_threshold_txt=self.cfg.prune_node_threshold_txt,
            edge_threshold_img=self.cfg.prune_edge_threshold_img,
            edge_threshold_txt=self.cfg.prune_edge_threshold_txt,
        )
        self.validator = LRMValidator(self.cfg)

    def capture(
        self,
        prompt: str,
        seed: int = 42,
        step: int = 0,
    ) -> FluxTrace:
        return self.capturer.capture(prompt, seed, step)

    def _build_config_metadata(self, trace: FluxTrace) -> Dict[str, Any]:
        return {
            "prompt": trace.prompt,
            "seed": trace.seed,
            "step": trace.step_idx,
            "timestep": trace.timestep,
            "model_id": self.cfg.model_id,
            "target_layers": list(self.cfg.target_layers),
            "height": self.cfg.height,
            "width": self.cfg.width,
            "S_img": trace.S_img,
            "S_txt": trace.S_txt,
            "circuit_max_nodes": self.cfg.circuit_max_nodes,
            "circuit_min_attribution": self.cfg.circuit_min_attribution,
            "prune_node_threshold_img": self.cfg.prune_node_threshold_img,
            "prune_node_threshold_txt": self.cfg.prune_node_threshold_txt,
            "prune_edge_threshold_img": self.cfg.prune_edge_threshold_img,
            "prune_edge_threshold_txt": self.cfg.prune_edge_threshold_txt,
        }

    def trace_circuit(
        self,
        trace: FluxTrace,
        layer: int,
        stream: str,
        position: int,
        feature_idx: int,
        expansion_cfg: Optional[ExpansionConfig] = None,
        prune: bool = True,
    ) -> Tuple[AggAttributionGraph, Dict[str, Any], Optional["AttributionGraph"]]:
        """Build a multi-hop circuit graph via iterative expansion."""
        tracer = self.circuit_tracer
        if expansion_cfg is not None:
            tracer = CircuitTracer(
                self.pipe.transformer,
                self.transcoders,
                self.cfg,
                expansion_cfg=expansion_cfg,
            )

        # Build per-position graph
        graph = tracer.trace_circuit(
            trace,
            layer,
            stream,
            position,
            feature_idx,
        )

        metrics = {}

        # Conservation check on the per-position graph.
        raw_val = self.validator.validate_attribution_sum(graph, verbose=False)
        metrics["raw_attribution_expected"] = raw_val["expected"]
        metrics["raw_attribution_actual"] = raw_val["actual"]
        metrics["raw_attribution_rel_err"] = raw_val["rel_error"]
        metrics["raw_num_nodes"] = len(graph.nodes)
        metrics["raw_num_edges"] = len(graph.edges)

        # Aggregate over positions, then re-check conservation.
        agg_graph = GraphAggregator.aggregate(graph, trace)
        agg_graph.metadata["config"] = self._build_config_metadata(trace)
        agg_val = self.validator.validate_attribution_sum(agg_graph, verbose=False)
        metrics["agg_attribution_actual"] = agg_val["actual"]
        metrics["agg_attribution_rel_err"] = agg_val["rel_error"]
        metrics["agg_num_nodes"] = len(agg_graph.nodes)
        metrics["agg_num_edges"] = len(agg_graph.edges)

        # Keep the per-position graph for perturbation validation.
        raw_graph = graph

        if prune and self.pruner is not None:
            agg_graph = self.pruner.prune(agg_graph)
            pruned_val = self.validator.validate_attribution_sum(
                agg_graph, verbose=False
            )
            metrics["pruned_attribution_expected"] = pruned_val["expected"]
            metrics["pruned_attribution_actual"] = pruned_val["actual"]
            metrics["pruned_attribution_rel_err"] = pruned_val["rel_error"]
            metrics["pruned_num_nodes"] = len(agg_graph.nodes)
            metrics["pruned_num_edges"] = len(agg_graph.edges)

        return agg_graph, metrics, raw_graph

    def validate_perturbation(
        self,
        trace: FluxTrace,
        graph: AggAttributionGraph,
        top_k: int = 30,
        plot: bool = True,
        pairwise: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        if pairwise:
            summary = self.perturbation_validator.validate(
                trace, graph, top_k=top_k, verbose=verbose
            )
        else:
            summary = self.perturbation_validator.validate_direct(
                trace, graph, top_k=top_k, verbose=verbose
            )
        if plot:
            self.perturbation_validator.plot_results(summary)
        return summary

    def run_experiment(
        self,
        targets: List[Dict[str, Any]],
        *,
        output_dir: str = "./attribution_results",
        expansion_cfg: Optional[ExpansionConfig] = None,
        perturbation_top_k: int = 30,
    ) -> "pd.DataFrame":
        """Full experiment pipeline. Returns a DataFrame with all metrics per target."""
        os.makedirs(output_dir, exist_ok=True)

        groups = defaultdict(list)
        for t in targets:
            prompt = t["prompt"]
            step = int(t.get("step", 0))
            seed = int(t.get("seed", 0))
            groups[(prompt, step, seed)].append(t)

        all_results = []
        target_num = 0
        total_targets = len(targets)

        for (prompt, step, seed), group in groups.items():
            trace = self.capture(prompt=prompt, seed=seed, step=step)

            for t in group:
                target_num += 1
                layer = int(t["layer"])
                stream = t["stream"]
                feat_idx = int(t["feat_idx"])

                pos, score = infer_position_for_feature(
                    trace,
                    layer=layer,
                    stream=stream,
                    feat_idx=feat_idx,
                )

                t0 = time()

                # Circuit tracing + aggregation + pruning + validation
                graph, metrics, raw_graph = self.trace_circuit(
                    trace,
                    layer=layer,
                    stream=stream,
                    position=pos,
                    feature_idx=feat_idx,
                    expansion_cfg=expansion_cfg,
                    prune=True,
                )

                elapsed = time() - t0
                metrics["elapsed_seconds"] = elapsed

                # Perturbation validation
                try:
                    pert_graph = raw_graph if raw_graph is not None else graph
                    pert_summary = self.validate_perturbation(
                        trace,
                        pert_graph,
                        top_k=perturbation_top_k,
                        plot=False,
                        pairwise=True,
                        verbose=False,
                    )
                    del raw_graph
                    metrics["spearman_r"] = pert_summary.get("spearman_r", float("nan"))
                    metrics["pearson_r"] = pert_summary.get("pearson_r", float("nan"))
                    metrics["n_nonzero_pairs"] = pert_summary.get("n_pairs", 0)
                except Exception as e:
                    print(
                        f"[{target_num}/{total_targets}] perturbation validation failed: {e}"
                    )
                    metrics["spearman_r"] = float("nan")
                    metrics["pearson_r"] = float("nan")
                    metrics["n_nonzero_pairs"] = 0

                # Save graph
                fname = (
                    f"circuit_f{feat_idx}_step{step}_seed{seed}"
                    f"_L{layer}_{stream}.json"
                )
                path = os.path.join(output_dir, fname)
                graph.save(path)
                metrics["graph_path"] = path

                row = {
                    "target_num": target_num,
                    "prompt": prompt,
                    "step": step,
                    "seed": seed,
                    "layer": layer,
                    "stream": stream,
                    "feat_idx": feat_idx,
                    "position": pos,
                    "preactivation": score,
                    **metrics,
                }
                all_results.append(row)

                print(
                    f"[{target_num}/{total_targets}] "
                    f"L{layer} {stream} f{feat_idx}  {elapsed:.0f}s  "
                    f"pruned {metrics.get('pruned_num_nodes', 0)} nodes / "
                    f"{metrics.get('pruned_num_edges', 0)} edges  "
                    f"rel_err {metrics.get('pruned_attribution_rel_err', 0):.2%}  "
                    f"spearman {metrics.get('spearman_r', float('nan')):.3f}"
                )

                del graph
                gc.collect()
                torch.cuda.empty_cache()

        df = pd.DataFrame(all_results)
        csv_path = os.path.join(output_dir, "experiment_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"{len(all_results)} targets processed -> {csv_path}")
        return df
