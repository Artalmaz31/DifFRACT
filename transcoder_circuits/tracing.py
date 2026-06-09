from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from .attribution_graph import NodeType, NodeId, EdgeData, AttributionGraph
from .edges import VJPComputer, EdgeComputer
from .influence import build_adjacency, normalized_adjacency, indirect_influence
from .replacement_model import FluxTrace, LRMConfig


@dataclass
class ExpansionConfig:
    max_nodes: int = 500
    min_attribution: float = 1e-4
    batch_size: int = 50
    verbose: bool = False


class CircuitTracer:
    def __init__(
        self,
        transformer: nn.Module,
        transcoders: Dict[str, nn.Module],
        cfg: LRMConfig,
        expansion_cfg: Optional[ExpansionConfig] = None,
    ):
        self.transformer = transformer
        self.transcoders = transcoders
        self.cfg = cfg
        self.expansion_cfg = expansion_cfg or ExpansionConfig(
            max_nodes=cfg.circuit_max_nodes,
            min_attribution=cfg.circuit_min_attribution,
            batch_size=cfg.expansion_batch_size,
        )

        self.vjp_computer = VJPComputer(transformer, transcoders, cfg)
        self.edge_computer = EdgeComputer(transcoders, cfg)

    def trace_circuit(
        self,
        trace: FluxTrace,
        target_layer: int,
        target_stream: str,
        target_position: int,
        target_feature_idx: int,
    ) -> AttributionGraph:
        ecfg = self.expansion_cfg

        target_id = NodeId(
            node_type=NodeType.FEATURE,
            layer=target_layer,
            stream=target_stream,
            position=target_position,
            feat_idx=target_feature_idx,
        )

        lc = trace.get_layer(target_layer)
        h_pre_tensor = getattr(lc, f"{target_stream}_h_pre", None)
        if h_pre_tensor is not None:
            target_preact = h_pre_tensor[0, target_position, target_feature_idx].item()
        else:
            z_tensor = getattr(lc, f"{target_stream}_z", None)
            target_preact = (
                z_tensor[0, target_position, target_feature_idx].item()
                if z_tensor is not None
                else 0.0
            )

        tc = self.transcoders[f"{target_stream}_{target_layer}"]
        block_shift = getattr(lc, f"{target_stream}_shift_mlp", None)
        encoder_bias = tc.get_effective_encoder_bias(
            target_feature_idx,
            trace.timestep_tensor,
            block_shift=block_shift,
        )

        graph = AttributionGraph(
            target_id=target_id,
            target_preactivation=target_preact,
            target_encoder_bias=encoder_bias,
        )
        graph.add_node(target_id, preactivation=target_preact)

        expanded: Set[NodeId] = set()
        discovered: Set[NodeId] = {target_id}

        if ecfg.verbose:
            print(f"Circuit tracing: target = {target_id}")

        self._expand_node(trace, graph, target_id, expanded, discovered)

        nodes_expanded = 1
        while nodes_expanded < ecfg.max_nodes:
            influences = self._compute_node_influences(graph, expanded)

            candidates = []
            for nid in discovered:
                if nid in expanded:
                    continue
                if nid.node_type != NodeType.FEATURE:
                    continue
                if nid.layer < self.cfg.first_layer:
                    continue
                inf = influences.get(nid, 0.0)
                candidates.append((inf, nid))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[0], reverse=True)
            batch = candidates[: ecfg.batch_size]

            for inf_score, node_id in batch:
                if nodes_expanded >= ecfg.max_nodes:
                    break

                self._expand_node(trace, graph, node_id, expanded, discovered)
                nodes_expanded += 1

                if ecfg.verbose and nodes_expanded % 100 == 0:
                    print(
                        f"  Expanded {nodes_expanded}/{ecfg.max_nodes} nodes | "
                        f"graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges | "
                        f"top influence: {inf_score:.6f}"
                    )

        if ecfg.verbose:
            print(
                f"  Pre-compaction: {len(graph.nodes)} nodes, {len(graph.edges)} edges"
            )
        graph = self._compact_graph(graph, expanded)
        if ecfg.verbose:
            print(
                f"  Post-compaction: {len(graph.nodes)} nodes, {len(graph.edges)} edges"
            )

        graph.metadata["circuit_tracing"] = {
            "nodes_expanded": nodes_expanded,
            "total_nodes": len(graph.nodes),
            "total_edges": len(graph.edges),
            "max_nodes_budget": ecfg.max_nodes,
        }

        if ecfg.verbose:
            feat_edges = sum(
                1 for e in graph.edges if e.source_id.node_type == NodeType.FEATURE
            )
            err_edges = sum(
                1 for e in graph.edges if e.source_id.node_type == NodeType.ERROR
            )
            res_edges = sum(
                1
                for e in graph.edges
                if e.source_id.node_type in (NodeType.RESIDUAL, NodeType.INPUT)
            )
            target_layers = sorted({e.target_id.layer for e in graph.edges})
            print(
                f"Circuit complete: {nodes_expanded} expanded, "
                f"{len(graph.nodes)} nodes, {len(graph.edges)} edges "
                f"(feat: {feat_edges}, err: {err_edges}, res: {res_edges})\n"
                f"  Edge target layers: {target_layers}"
            )

        return graph

    @staticmethod
    def _compute_attn_error_bias(
        trace: FluxTrace,
        target_node: NodeId,
        vjp_grads: Dict[Tuple[int, str], Optional[Tensor]],
    ) -> float:
        """Compute the total contribution of frozen attention errors to h_pre."""
        total = 0.0
        for layer in sorted(trace.layer_caches.keys()):
            if layer >= target_node.layer:
                continue
            lc = trace.get_layer(layer)
            ac = lc.attention
            if ac is None:
                continue
            for stream in ("img", "txt"):
                gate_msa = getattr(lc, f"{stream}_gate_msa", None)
                attn_err = getattr(ac, f"attn_error_{stream}", None)
                if gate_msa is None or attn_err is None:
                    continue
                write_layer = min(layer + 1, target_node.layer)
                vjp_grad = vjp_grads.get((write_layer, stream))
                if vjp_grad is None:
                    continue
                device = vjp_grad.device
                gate = gate_msa.to(device=device, dtype=torch.float32)
                err = attn_err.to(device=device, dtype=torch.float32)
                grad = vjp_grad.to(dtype=torch.float32)
                if gate.dim() == 2:
                    gate = gate.unsqueeze(1)
                gated_err = gate * err
                total += (gated_err * grad).sum().item()
        return total

    @staticmethod
    def _compute_decoder_bias_contribution(
        trace: FluxTrace,
        target_node: NodeId,
        vjp_grads: Dict[Tuple[int, str], Optional[Tensor]],
        transcoders: Dict[str, nn.Module],
    ) -> float:
        """Compute total contribution of decoder biases to h_pre."""
        total = 0.0
        for layer in sorted(trace.layer_caches.keys()):
            if layer >= target_node.layer:
                continue
            lc = trace.get_layer(layer)
            for stream in ("img", "txt"):
                gate_mlp = getattr(lc, f"{stream}_gate_mlp", None)
                if gate_mlp is None:
                    continue
                tc_key = f"{stream}_{layer}"
                tc = transcoders.get(tc_key)
                if tc is None:
                    continue
                b_dec = tc.decoder.bias
                if b_dec is None:
                    continue
                write_layer = min(layer + 1, target_node.layer)
                vjp_grad = vjp_grads.get((write_layer, stream))
                if vjp_grad is None:
                    continue
                device = vjp_grad.device
                gate = gate_mlp.to(device=device, dtype=torch.float32)
                bias = b_dec.to(device=device, dtype=torch.float32)
                grad = vjp_grad.to(dtype=torch.float32)
                if gate.dim() == 2:
                    gate = gate.unsqueeze(1)
                gated_bias = gate * bias
                total += (gated_bias * grad).sum().item()
        return total

    def _expand_node(
        self,
        trace: FluxTrace,
        graph: AttributionGraph,
        target_node: NodeId,
        expanded: Set[NodeId],
        discovered: Set[NodeId],
    ) -> None:
        ecfg = self.expansion_cfg

        vjp_grads = self.vjp_computer.compute_feature_vjp(
            trace,
            target_node.layer,
            target_node.stream,
            target_node.position,
            target_node.feat_idx,
        )

        if target_node == graph.target_id:
            attn_err_bias = self._compute_attn_error_bias(
                trace,
                target_node,
                vjp_grads,
            )
            dec_bias_contrib = self._compute_decoder_bias_contribution(
                trace,
                target_node,
                vjp_grads,
                self.transcoders,
            )
            graph.target_encoder_bias += attn_err_bias + dec_bias_contrib

        all_edges = self.edge_computer.compute_feature_edges(
            trace,
            target_node,
            vjp_grads,
        )

        all_edges.sort(key=lambda e: abs(e.attribution), reverse=True)

        kept = 0
        for edge in all_edges:
            if abs(edge.attribution) < ecfg.min_attribution:
                break
            graph.add_edge(edge)
            kept += 1

            src = edge.source_id
            if src not in discovered:
                discovered.add(src)

        expanded.add(target_node)

    def _compute_node_influences(
        self,
        graph: AttributionGraph,
        expanded: Set[NodeId],
    ) -> Dict[NodeId, float]:
        expanded_list = sorted(expanded)
        node_to_idx = {nid: i for i, nid in enumerate(expanded_list)}
        N = len(expanded_list)

        if N == 0:
            return {}

        A = build_adjacency(graph.edges, node_to_idx)
        A_norm = normalized_adjacency(A)
        B = indirect_influence(A_norm)

        target_idx = node_to_idx.get(graph.target_id)
        w = np.zeros(N, dtype=np.float64)
        if target_idx is not None:
            w[target_idx] = 1.0

        influence_vec = B @ w

        reach = w + influence_vec

        result: Dict[NodeId, float] = {}
        for nid, idx in node_to_idx.items():
            result[nid] = float(influence_vec[idx])

        for edge in graph.edges:
            src = edge.source_id
            if src in expanded:
                continue
            tgt_idx = node_to_idx.get(edge.target_id)
            if tgt_idx is None:
                continue
            if src not in result:
                result[src] = 0.0
            result[src] += abs(edge.attribution) * float(reach[tgt_idx])

        return result

    def _compact_graph(
        self,
        graph: AttributionGraph,
        expanded: Set[NodeId],
    ) -> AttributionGraph:
        compact = AttributionGraph(
            target_id=graph.target_id,
            target_preactivation=graph.target_preactivation,
            target_encoder_bias=graph.target_encoder_bias,
            metadata=dict(graph.metadata),
        )

        for nid in expanded:
            key = str(nid)
            if key in graph.nodes:
                compact.nodes[key] = graph.nodes[key]
            else:
                compact.add_node(nid)

        unexplored_agg: Dict[Tuple, float] = {}

        for edge in graph.edges:
            src = edge.source_id
            tgt = edge.target_id

            if src in expanded and tgt in expanded:
                compact.add_edge(edge)
            elif src.node_type in (NodeType.ERROR, NodeType.RESIDUAL, NodeType.INPUT):
                if tgt in expanded:
                    compact.add_edge(edge)
            elif src.node_type == NodeType.FEATURE and tgt in expanded:
                agg_key = (tgt, src.layer, src.stream, src.position)
                unexplored_agg[agg_key] = (
                    unexplored_agg.get(agg_key, 0.0) + edge.attribution
                )

        thresh = self.expansion_cfg.min_attribution
        for (tgt, layer, stream, position), total_attr in unexplored_agg.items():
            if abs(total_attr) < thresh:
                continue
            agg_source = NodeId(
                node_type=NodeType.ERROR,
                layer=layer,
                stream=stream,
                position=position,
                feat_idx=-2,
            )
            compact.add_edge(
                EdgeData.for_error(
                    source=agg_source,
                    target=tgt,
                    error_contrib=total_attr,
                )
            )

        return compact
