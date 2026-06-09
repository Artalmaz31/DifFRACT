"""Indirect-influence linear algebra shared across tracing, pruning and validation."""

from typing import Dict, List, Tuple

import numpy as np


def build_node_index(graph) -> Tuple[List, Dict]:
    """Collect every node referenced by the graph and assign it a matrix index."""
    node_list: List = []
    node_to_idx: Dict = {}

    def _add(nid):
        if nid not in node_to_idx:
            node_to_idx[nid] = len(node_list)
            node_list.append(nid)

    for info in graph.nodes.values():
        _add(info["id"])
    for edge in graph.edges:
        _add(edge.source_id)
        _add(edge.target_id)

    return node_list, node_to_idx


def build_adjacency(edges, node_to_idx: Dict) -> np.ndarray:
    """Adjacency matrix A[i, j] = summed attribution from node i to node j."""
    n = len(node_to_idx)
    A = np.zeros((n, n), dtype=np.float64)
    for edge in edges:
        src = node_to_idx.get(edge.source_id)
        tgt = node_to_idx.get(edge.target_id)
        if src is not None and tgt is not None:
            A[src, tgt] += edge.attribution
    return A


def normalized_adjacency(A: np.ndarray) -> np.ndarray:
    """Column-normalise |A| so each node's incoming weights sum to one."""
    A_abs = np.abs(A)
    col_sums = np.maximum(A_abs.sum(axis=0), 1e-8)
    return A_abs / col_sums[None, :]


def indirect_influence(A_norm: np.ndarray) -> np.ndarray:
    """B = (I - A_norm)^{-1} - I — total influence over all paths."""
    n = A_norm.shape[0]
    I = np.eye(n, dtype=np.float64)
    try:
        return np.linalg.inv(I - A_norm) - I
    except np.linalg.LinAlgError:
        return np.linalg.pinv(I - A_norm) - I
