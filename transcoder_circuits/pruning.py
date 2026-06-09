from copy import deepcopy
from typing import Dict, List, Set, Tuple
import numpy as np
from .attribution_graph import AggNodeId, AggEdgeData, AggAttributionGraph
from .influence import (
    build_node_index,
    build_adjacency,
    normalized_adjacency,
    indirect_influence,
)


class GraphPruner:
    """Two-step pruning of an AggAttributionGraph."""

    def __init__(
        self,
        node_threshold_img: float = 0.8,
        node_threshold_txt: float = 0.8,
        edge_threshold_img: float = 0.98,
        edge_threshold_txt: float = 0.98,
    ):
        self.node_threshold_img = node_threshold_img
        self.node_threshold_txt = node_threshold_txt
        self.edge_threshold_img = edge_threshold_img
        self.edge_threshold_txt = edge_threshold_txt

    def prune(self, graph: "AggAttributionGraph") -> "AggAttributionGraph":
        """Run the full pruning pipeline and return a new pruned graph."""
        pruned = self._prune_nodes(graph)
        pruned = self._prune_edges(pruned)

        pruned.metadata["pruning"] = {
            "node_threshold_img": self.node_threshold_img,
            "node_threshold_txt": self.node_threshold_txt,
            "edge_threshold_img": self.edge_threshold_img,
            "edge_threshold_txt": self.edge_threshold_txt,
            "original_num_nodes": len(graph.nodes),
            "original_num_edges": len(graph.edges),
            "pruned_num_nodes": len(pruned.nodes),
            "pruned_num_edges": len(pruned.edges),
            "original_attribution_sum": graph.get_attribution_sum(),
            "pruned_attribution_sum": pruned.get_attribution_sum(),
        }

        return pruned

    def _prune_nodes(
        self,
        graph: "AggAttributionGraph",
    ) -> "AggAttributionGraph":
        """Remove feature nodes whose indirect influence on the target is small."""
        node_list, node_to_idx = build_node_index(graph)
        if len(node_list) == 0:
            return graph

        A_norm = normalized_adjacency(build_adjacency(graph.edges, node_to_idx))
        B = indirect_influence(A_norm)

        # Target node weight vector
        logit_weights = self._get_target_weights(graph, node_list, node_to_idx)

        # Influence of each node on the target
        influence = B @ logit_weights

        # Determine which nodes to keep
        keep_set = self._select_nodes_by_cumulative_influence(
            node_list, influence, graph.target_id
        )

        # Build pruned graph
        return self._create_subgraph(graph, keep_set)

    def _prune_edges(
        self,
        graph: "AggAttributionGraph",
    ) -> "AggAttributionGraph":
        """Remove edges with small influence on the target."""
        node_list, node_to_idx = build_node_index(graph)
        if len(node_list) == 0 or len(graph.edges) == 0:
            return graph

        A_norm = normalized_adjacency(build_adjacency(graph.edges, node_to_idx))
        B = indirect_influence(A_norm)
        logit_weights = self._get_target_weights(graph, node_list, node_to_idx)

        # Node influence scores
        node_score = B @ logit_weights
        target_idx = node_to_idx.get(graph.target_id)
        if target_idx is not None:
            node_score[target_idx] = 1.0

        # Edge score for edge i->j = A_norm[i,j] * node_score[j]
        edge_score_matrix = A_norm * node_score[None, :]

        # Determine which edges to keep
        img_edges_with_score = []
        txt_edges_with_score = []

        for edge in graph.edges:
            src_idx = node_to_idx.get(edge.source_id)
            tgt_idx = node_to_idx.get(edge.target_id)
            if src_idx is None or tgt_idx is None:
                continue
            score = edge_score_matrix[src_idx, tgt_idx]
            if edge.source_id.stream == "img":
                img_edges_with_score.append((edge, score))
            else:
                txt_edges_with_score.append((edge, score))

        keep_edges = self._select_edges_by_cumulative_score(
            img_edges_with_score, self.edge_threshold_img
        ) + self._select_edges_by_cumulative_score(
            txt_edges_with_score, self.edge_threshold_txt
        )

        return self._create_graph_with_edges(graph, keep_edges)

    @staticmethod
    def _select_edges_by_cumulative_score(
        edges_with_score: List[Tuple["AggEdgeData", float]],
        threshold: float,
    ) -> List["AggEdgeData"]:
        """Keep edges with highest score until cumulative >= threshold."""
        if not edges_with_score:
            return []
        edges_with_score.sort(key=lambda x: x[1], reverse=True)
        total = sum(s for _, s in edges_with_score)
        if total < 1e-12:
            return [e for e, _ in edges_with_score]
        keep = []
        cumulative = 0.0
        for edge, score in edges_with_score:
            cumulative += score
            keep.append(edge)
            if cumulative / total >= threshold:
                break
        return keep

    @staticmethod
    def _get_target_weights(
        graph: "AggAttributionGraph",
        node_list: List["AggNodeId"],
        node_to_idx: Dict["AggNodeId", int],
    ) -> np.ndarray:
        N = len(node_list)
        weights = np.zeros(N, dtype=np.float64)
        idx = node_to_idx.get(graph.target_id)
        if idx is not None:
            weights[idx] = 1.0
        return weights

    def _select_nodes_by_cumulative_influence(
        self,
        node_list: List["AggNodeId"],
        influence: np.ndarray,
        target_id: "AggNodeId",
    ) -> Set["AggNodeId"]:
        keep: Set["AggNodeId"] = set()
        prunable_img: List[Tuple[int, float]] = []
        prunable_txt: List[Tuple[int, float]] = []

        for idx, nid in enumerate(node_list):
            if nid == target_id:
                keep.add(nid)
            elif nid.node_type.value in ("error", "residual", "input"):
                # Error and input vertices are exempt from pruning
                keep.add(nid)
            elif nid.stream == "img":
                prunable_img.append((idx, influence[idx]))
            else:
                prunable_txt.append((idx, influence[idx]))

        keep |= self._select_from_prunable(
            prunable_img, node_list, self.node_threshold_img
        )
        keep |= self._select_from_prunable(
            prunable_txt, node_list, self.node_threshold_txt
        )

        return keep

    @staticmethod
    def _select_from_prunable(
        prunable: List[Tuple[int, float]],
        node_list: List["AggNodeId"],
        threshold: float,
    ) -> Set["AggNodeId"]:
        if not prunable:
            return set()
        prunable.sort(key=lambda x: x[1], reverse=True)
        total = sum(inf for _, inf in prunable)
        if total < 1e-12:
            return set()
        keep = set()
        cumulative = 0.0
        for orig_idx, inf in prunable:
            cumulative += inf
            keep.add(node_list[orig_idx])
            if cumulative / total >= threshold:
                break
        return keep

    @staticmethod
    def _create_subgraph(
        graph: "AggAttributionGraph",
        keep_nodes: Set["AggNodeId"],
    ) -> "AggAttributionGraph":
        new_graph = AggAttributionGraph(
            target_id=graph.target_id,
            target_position=graph.target_position,
            target_preactivation=graph.target_preactivation,
            target_encoder_bias=graph.target_encoder_bias,
            metadata=deepcopy(graph.metadata),
        )

        for key, info in graph.nodes.items():
            nid = info["id"]
            if nid in keep_nodes:
                new_graph.nodes[key] = info
                if key in graph.activation_maps:
                    new_graph.activation_maps[key] = graph.activation_maps[key]

        for edge in graph.edges:
            if edge.source_id in keep_nodes and edge.target_id in keep_nodes:
                new_graph.edges.append(edge)

        return new_graph

    @staticmethod
    def _create_graph_with_edges(
        graph: "AggAttributionGraph",
        edges: List["AggEdgeData"],
    ) -> "AggAttributionGraph":
        referenced: Set["AggNodeId"] = set()
        for edge in edges:
            referenced.add(edge.source_id)
            referenced.add(edge.target_id)

        referenced.add(graph.target_id)
        for key, info in graph.nodes.items():
            nid = info["id"]
            if nid.node_type.value in ("error", "residual", "input"):
                referenced.add(nid)

        new_graph = AggAttributionGraph(
            target_id=graph.target_id,
            target_position=graph.target_position,
            target_preactivation=graph.target_preactivation,
            target_encoder_bias=graph.target_encoder_bias,
            metadata=deepcopy(graph.metadata),
        )

        for key, info in graph.nodes.items():
            nid = info["id"]
            if nid in referenced:
                new_graph.nodes[key] = info
                if key in graph.activation_maps:
                    new_graph.activation_maps[key] = graph.activation_maps[key]

        new_graph.edges = list(edges)
        return new_graph
