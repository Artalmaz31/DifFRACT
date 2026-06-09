from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple
import json
from .replacement_model import FluxTrace


class NodeType(Enum):
    OUTPUT = "output"
    FEATURE = "feature"
    ERROR = "error"
    RESIDUAL = "residual"
    INPUT = "input"


@dataclass(frozen=True)
class NodeId:
    node_type: NodeType
    layer: int
    stream: str
    position: int
    feat_idx: int = -1

    def __repr__(self) -> str:
        base = f"{self.node_type.value}_L{self.layer}_{self.stream}_p{self.position}"
        if self.feat_idx >= 0:
            base += f"_f{self.feat_idx}"
        return base

    def __hash__(self):
        return hash(
            (self.node_type, self.layer, self.stream, self.position, self.feat_idx)
        )

    def __lt__(self, other):
        return (
            self.node_type.value,
            self.layer,
            self.stream,
            self.position,
            self.feat_idx,
        ) < (
            other.node_type.value,
            other.layer,
            other.stream,
            other.position,
            other.feat_idx,
        )


@dataclass
class EdgeData:
    source_id: NodeId
    target_id: NodeId
    activation: float
    weight: float
    attribution: float

    @classmethod
    def for_feature(cls, source: NodeId, target: NodeId, z_src: float, w_eff: float):
        return cls(
            source_id=source,
            target_id=target,
            activation=z_src,
            weight=w_eff,
            attribution=z_src * w_eff,
        )

    @classmethod
    def for_error(cls, source: NodeId, target: NodeId, error_contrib: float):
        return cls(
            source_id=source,
            target_id=target,
            activation=1.0,
            weight=error_contrib,
            attribution=error_contrib,
        )

    @classmethod
    def for_residual(cls, source: NodeId, target: NodeId, residual_contrib: float):
        return cls(
            source_id=source,
            target_id=target,
            activation=1.0,
            weight=residual_contrib,
            attribution=residual_contrib,
        )


@dataclass
class AttributionGraph:
    target_id: NodeId
    target_preactivation: float = 0.0
    target_encoder_bias: float = 0.0
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: List[EdgeData] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node_id: NodeId, **kwargs):
        key = str(node_id)
        self.nodes[key] = {"id": node_id, "type": node_id.node_type.value, **kwargs}

    def add_edge(self, edge: EdgeData):
        self.edges.append(edge)
        src_key = str(edge.source_id)
        if src_key not in self.nodes:
            self.add_node(edge.source_id)

    def get_attribution_sum(self) -> float:
        return sum(e.attribution for e in self.edges)

    def to_dict(self) -> Dict:
        return {
            "target": str(self.target_id),
            "target_preactivation": self.target_preactivation,
            "target_encoder_bias": self.target_encoder_bias,
            "attribution_sum": self.get_attribution_sum(),
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "edges": [
                {
                    "src": str(e.source_id),
                    "tgt": str(e.target_id),
                    "activation": e.activation,
                    "weight": e.weight,
                    "attribution": e.attribution,
                }
                for e in self.edges
            ],
            "metadata": self.metadata,
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass(frozen=True)
class AggNodeId:
    """Position-independent node identifier."""

    node_type: NodeType
    layer: int
    stream: str
    feat_idx: int = -1

    def __repr__(self) -> str:
        if self.node_type == NodeType.INPUT:
            return f"input_{self.stream}"
        base = f"{self.node_type.value}_L{self.layer}_{self.stream}"
        if self.feat_idx >= 0:
            base += f"_f{self.feat_idx}"
        return base

    def __hash__(self):
        return hash((self.node_type, self.layer, self.stream, self.feat_idx))

    def __lt__(self, other):
        return (self.node_type.value, self.layer, self.stream, self.feat_idx) < (
            other.node_type.value,
            other.layer,
            other.stream,
            other.feat_idx,
        )


@dataclass
class AggEdgeData:
    """Edge in the position-aggregated attribution graph."""

    source_id: AggNodeId
    target_id: AggNodeId
    attribution: float


@dataclass
class AggAttributionGraph:
    """Position-aggregated attribution graph."""

    target_id: AggNodeId
    target_position: int = 0
    target_preactivation: float = 0.0
    target_encoder_bias: float = 0.0
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: List[AggEdgeData] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    activation_maps: Dict[str, Dict[int, float]] = field(default_factory=dict)

    def add_node(self, node_id: AggNodeId, **kwargs):
        key = str(node_id)
        self.nodes[key] = {
            "id": node_id,
            "type": node_id.node_type.value,
            "layer": node_id.layer,
            "stream": node_id.stream,
            "feat_idx": node_id.feat_idx,
            **kwargs,
        }

    def add_edge(self, edge: AggEdgeData):
        self.edges.append(edge)
        src_key = str(edge.source_id)
        if src_key not in self.nodes:
            self.add_node(edge.source_id)

    def get_attribution_sum(self) -> float:
        return sum(e.attribution for e in self.edges if e.target_id == self.target_id)

    def to_dict(self) -> Dict:
        target_info = {
            "id": str(self.target_id),
            "layer": self.target_id.layer,
            "stream": self.target_id.stream,
            "feat_idx": self.target_id.feat_idx,
            "position": self.target_position,
            "preactivation": self.target_preactivation,
            "encoder_bias": self.target_encoder_bias,
        }

        num_feat = sum(1 for v in self.nodes.values() if v["type"] == "feature")
        num_err = sum(1 for v in self.nodes.values() if v["type"] == "error")
        num_res = sum(1 for v in self.nodes.values() if v["type"] == "residual")
        num_inp = sum(1 for v in self.nodes.values() if v["type"] == "input")
        nodes_img = sum(1 for v in self.nodes.values() if v.get("stream") == "img")
        nodes_txt = sum(1 for v in self.nodes.values() if v.get("stream") == "txt")
        edges_feat = sum(
            1 for e in self.edges if e.source_id.node_type == NodeType.FEATURE
        )
        edges_err = sum(
            1 for e in self.edges if e.source_id.node_type == NodeType.ERROR
        )
        edges_res = sum(
            1
            for e in self.edges
            if e.source_id.node_type in (NodeType.RESIDUAL, NodeType.INPUT)
        )

        stats = {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "num_feature_nodes": num_feat,
            "num_error_nodes": num_err,
            "num_residual_nodes": num_res,
            "num_input_nodes": num_inp,
            "attribution_sum": self.get_attribution_sum(),
            "nodes_by_stream": {"img": nodes_img, "txt": nodes_txt},
            "edges_by_type": {
                "feature": edges_feat,
                "error": edges_err,
                "residual": edges_res,
                "input": sum(
                    1 for e in self.edges if e.source_id.node_type == NodeType.INPUT
                ),
            },
        }

        nodes_list = []
        for key, info in self.nodes.items():
            node_dict = {
                "id": key,
                "type": info["type"],
                "layer": info.get("layer", -1),
                "stream": info.get("stream", ""),
                "feat_idx": info.get("feat_idx", -1),
            }
            amap = self.activation_maps.get(key, {})
            if amap:
                node_dict["activation_map"] = {
                    str(p): round(v, 6) for p, v in amap.items()
                }
            nodes_list.append(node_dict)

        edges_list = [
            {
                "src": str(e.source_id),
                "tgt": str(e.target_id),
                "attribution": round(e.attribution, 8),
            }
            for e in self.edges
        ]

        config = self.metadata.get("config", {})

        return {
            "version": 2,
            "target": target_info,
            "config": config,
            "stats": stats,
            "nodes": nodes_list,
            "edges": edges_list,
            "metadata": {k: v for k, v in self.metadata.items() if k != "config"},
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class GraphAggregator:
    """Aggregates a per-position AttributionGraph into a position-independent AggAttributionGraph."""

    @staticmethod
    def _to_agg(nid: NodeId) -> AggNodeId:
        return AggNodeId(
            node_type=nid.node_type,
            layer=nid.layer,
            stream=nid.stream,
            feat_idx=nid.feat_idx,
        )

    @staticmethod
    def aggregate(
        graph: "AttributionGraph",
        trace: "FluxTrace",
    ) -> AggAttributionGraph:
        target_agg = GraphAggregator._to_agg(graph.target_id)

        agg = AggAttributionGraph(
            target_id=target_agg,
            target_position=graph.target_id.position,
            target_preactivation=graph.target_preactivation,
            target_encoder_bias=graph.target_encoder_bias,
            metadata=dict(graph.metadata),
        )
        agg.add_node(target_agg, preactivation=graph.target_preactivation)

        # 1. Aggregate edges
        edge_agg: Dict[Tuple, float] = {}
        for edge in graph.edges:
            src_agg = GraphAggregator._to_agg(edge.source_id)
            tgt_agg = GraphAggregator._to_agg(edge.target_id)
            key = (src_agg, tgt_agg)
            edge_agg[key] = edge_agg.get(key, 0.0) + edge.attribution

        for (src_agg, tgt_agg), attr in edge_agg.items():
            agg.add_edge(
                AggEdgeData(
                    source_id=src_agg,
                    target_id=tgt_agg,
                    attribution=attr,
                )
            )

        # 2. Build activation maps
        feature_ids: Set[Tuple[int, str, int]] = set()
        for key, info in agg.nodes.items():
            nid = info["id"]
            if nid.node_type == NodeType.FEATURE and nid.feat_idx >= 0:
                feature_ids.add((nid.layer, nid.stream, nid.feat_idx))

        for layer, stream, feat_idx in feature_ids:
            agg_id = AggNodeId(NodeType.FEATURE, layer, stream, feat_idx)
            agg_key = str(agg_id)
            lc = trace.get_layer(layer)
            z_tensor = getattr(lc, f"{stream}_z", None)
            if z_tensor is not None:
                z_feat = z_tensor[0, :, feat_idx].float().cpu()
                act_map = {}
                for pos in range(z_feat.shape[0]):
                    val = z_feat[pos].item()
                    if abs(val) > 1e-8:
                        act_map[pos] = val
                if act_map:
                    agg.activation_maps[agg_key] = act_map

        err_res_maps: Dict[str, Dict[int, float]] = {}
        for edge in graph.edges:
            src = edge.source_id
            if src.node_type in (NodeType.ERROR, NodeType.RESIDUAL, NodeType.INPUT):
                agg_key = str(GraphAggregator._to_agg(src))
                if agg_key not in err_res_maps:
                    err_res_maps[agg_key] = {}
                pos = src.position
                err_res_maps[agg_key][pos] = (
                    err_res_maps[agg_key].get(pos, 0.0) + edge.attribution
                )
        for agg_key, pmap in err_res_maps.items():
            agg.activation_maps[agg_key] = pmap

        target_lc = trace.get_layer(graph.target_id.layer)
        z_target = getattr(target_lc, f"{graph.target_id.stream}_z", None)
        if z_target is not None:
            target_z_val = z_target[
                0, graph.target_id.position, graph.target_id.feat_idx
            ].item()
            agg.activation_maps[str(target_agg)] = {
                graph.target_id.position: target_z_val,
            }

        return agg
