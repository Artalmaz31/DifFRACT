from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from .attribution_graph import NodeType, NodeId, EdgeData
from .replacement_model import FluxTrace, LRMConfig, LRMPatcher


class VJPComputer:
    def __init__(
        self,
        transformer: nn.Module,
        transcoders: Dict[str, nn.Module],
        cfg: "LRMConfig",
    ):
        self.transformer = transformer
        self.transcoders = transcoders
        self.cfg = cfg

    def compute_feature_vjp(
        self,
        trace: FluxTrace,
        target_layer: int,
        target_stream: str,
        target_position: int,
        target_feature_idx: int,
    ) -> Dict[Tuple[int, str], Optional[Tensor]]:
        device = self.cfg.device
        boundary_img = trace.boundary_img.to(device).clone().requires_grad_(True)
        boundary_txt = trace.boundary_txt.to(device).clone().requires_grad_(True)

        layer_inputs: Dict[int, Dict[str, Tensor]] = {}
        x_ff_container: Dict[str, Optional[Tensor]] = {"value": None}
        boundary_injected = {"done": False}

        hooks = []
        try:

            def boundary_inject_hook(module, args, kwargs):
                if boundary_injected["done"]:
                    return None

                boundary_injected["done"] = True
                new_args = list(args)
                new_kwargs = dict(kwargs)

                if "hidden_states" in new_kwargs:
                    new_kwargs["hidden_states"] = boundary_img
                if "encoder_hidden_states" in new_kwargs:
                    new_kwargs["encoder_hidden_states"] = boundary_txt

                if len(new_args) >= 1:
                    new_args[0] = boundary_img
                if len(new_args) >= 2:
                    new_args[1] = boundary_txt

                return (tuple(new_args), new_kwargs)

            def make_layer_input_hook(layer_idx: int):
                def hook(module, args, kwargs):
                    hs = kwargs.get("hidden_states", args[0] if args else None)
                    ehs = kwargs.get(
                        "encoder_hidden_states", args[1] if len(args) > 1 else None
                    )
                    layer_inputs[layer_idx] = {"img": hs, "txt": ehs}

                return hook

            def target_ff_pre_hook(module, args):
                x_ff_container["value"] = args[0]

            first_blk = self.transformer.transformer_blocks[self.cfg.first_layer]
            h = first_blk.register_forward_pre_hook(
                boundary_inject_hook, with_kwargs=True
            )
            hooks.append(h)

            for layer_idx in self.cfg.target_layers:
                blk = self.transformer.transformer_blocks[layer_idx]
                h = blk.register_forward_pre_hook(
                    make_layer_input_hook(layer_idx), with_kwargs=True
                )
                hooks.append(h)

            with LRMPatcher(
                self.transformer, trace, self.transcoders, self.cfg, mode="linear"
            ):
                target_blk = self.transformer.transformer_blocks[target_layer]
                target_ff = (
                    target_blk.ff if target_stream == "img" else target_blk.ff_context
                )

                h = target_ff.register_forward_pre_hook(target_ff_pre_hook)
                hooks.append(h)

                kwargs = dict(trace.transformer_kwargs)
                _ = self.transformer(**kwargs)

            x_ff = x_ff_container["value"]
            tc_key = f"{target_stream}_{target_layer}"
            tc = self.transcoders[tc_key]

            tc.to(device)
            tc_dtype = next(tc.parameters()).dtype
            x_ff_tc = x_ff.to(dtype=tc_dtype)

            t = trace.timestep_tensor.to(device=device, dtype=torch.float32).view(-1)
            h_pre = tc.feature_preactivation(x_ff_tc, t, target_feature_idx)[
                0, target_position
            ]

            grad_inputs = []
            grad_keys = []

            for layer_idx in sorted(self.cfg.target_layers):
                if layer_idx in layer_inputs:
                    if layer_inputs[layer_idx]["img"] is not None:
                        grad_inputs.append(layer_inputs[layer_idx]["img"])
                        grad_keys.append((layer_idx, "img"))
                    if layer_inputs[layer_idx]["txt"] is not None:
                        grad_inputs.append(layer_inputs[layer_idx]["txt"])
                        grad_keys.append((layer_idx, "txt"))

            grad_inputs.extend([boundary_img, boundary_txt])
            grad_keys.extend(
                [
                    (self.cfg.first_layer - 1, "img"),
                    (self.cfg.first_layer - 1, "txt"),
                ]
            )

            grads = torch.autograd.grad(
                h_pre,
                grad_inputs,
                allow_unused=True,
                retain_graph=False,
            )

            sensitivities = {}
            for key, grad in zip(grad_keys, grads):
                sensitivities[key] = grad.detach() if grad is not None else None

            return sensitivities
        finally:
            tc_key = f"{target_stream}_{target_layer}"
            self.transcoders[tc_key].cpu()
            for h in hooks:
                h.remove()


class EdgeComputer:
    def __init__(
        self,
        transcoders: Dict[str, nn.Module],
        cfg: "LRMConfig",
    ):
        self.transcoders = transcoders
        self.cfg = cfg

    def compute_feature_edges(
        self,
        trace: FluxTrace,
        target_node: NodeId,
        vjp_grads: Dict[Tuple[int, str], Optional[Tensor]],
    ) -> List[EdgeData]:
        edges = []

        for src_layer in range(self.cfg.first_layer, target_node.layer):
            for src_stream in ["img", "txt"]:
                src_edges = self._compute_feature_edges_from_layer(
                    trace, src_layer, src_stream, target_node, vjp_grads
                )
                edges.extend(src_edges)

        for src_layer in range(self.cfg.first_layer, target_node.layer):
            for src_stream in ["img", "txt"]:
                err_edges = self._compute_error_edges_per_position(
                    trace, src_layer, src_stream, target_node, vjp_grads
                )
                edges.extend(err_edges)

        boundary_edges = self._compute_boundary_edges_per_position(
            trace, target_node, vjp_grads
        )
        edges.extend(boundary_edges)

        return edges

    def _compute_feature_edges_from_layer(
        self,
        trace: FluxTrace,
        src_layer: int,
        src_stream: str,
        target_node: NodeId,
        vjp_grads: Dict[Tuple[int, str], Optional[Tensor]],
    ) -> List[EdgeData]:
        edges = []
        device = self.cfg.device
        lc = trace.get_layer(src_layer)

        z_src = getattr(lc, f"{src_stream}_z", None)
        if z_src is None:
            return edges
        z_src = z_src.to(device, dtype=torch.float32)

        write_layer = min(src_layer + 1, target_node.layer)
        v = vjp_grads.get((write_layer, src_stream))
        if v is None:
            return edges
        v = v.to(device, dtype=torch.float32)

        gate = getattr(lc, f"{src_stream}_gate_mlp", None)
        if gate is not None:
            gate = gate.to(device, dtype=torch.float32)
            if gate.dim() == 2:
                gate = gate.unsqueeze(1)
        else:
            gate = torch.ones(1, 1, v.shape[-1], device=device, dtype=torch.float32)

        tc_src = self.transcoders.get(f"{src_stream}_{src_layer}")
        if tc_src is None:
            return edges
        W_dec = tc_src.decoder.weight.to(device, dtype=torch.float32)

        z_flat = z_src[0]
        v_flat = v[0]
        gate_vec = gate[0, 0]

        v_gated = v_flat * gate_vec
        w_eff = torch.matmul(v_gated, W_dec)
        attributions = z_flat * w_eff

        thresh = self.cfg.circuit_min_attribution
        z_active = z_flat.abs() > thresh
        attr_above = attributions.abs() >= thresh
        valid_mask = z_active & attr_above

        coords = torch.nonzero(valid_mask, as_tuple=False)
        if coords.shape[0] == 0:
            return edges

        pos_indices = coords[:, 0]
        feat_indices = coords[:, 1]
        z_vals = z_flat[pos_indices, feat_indices]
        w_vals = w_eff[pos_indices, feat_indices]

        pos_cpu = pos_indices.cpu().numpy()
        feat_cpu = feat_indices.cpu().numpy()
        z_cpu = z_vals.cpu().numpy()
        w_cpu = w_vals.cpu().numpy()

        for idx in range(len(pos_cpu)):
            source_id = NodeId(
                node_type=NodeType.FEATURE,
                layer=src_layer,
                stream=src_stream,
                position=int(pos_cpu[idx]),
                feat_idx=int(feat_cpu[idx]),
            )
            edges.append(
                EdgeData.for_feature(
                    source=source_id,
                    target=target_node,
                    z_src=float(z_cpu[idx]),
                    w_eff=float(w_cpu[idx]),
                )
            )

        return edges

    def _compute_error_edges_per_position(
        self,
        trace: FluxTrace,
        src_layer: int,
        src_stream: str,
        target_node: NodeId,
        vjp_grads: Dict[Tuple[int, str], Optional[Tensor]],
    ) -> List[EdgeData]:
        edges = []
        device = self.cfg.device

        lc = trace.get_layer(src_layer)

        error = getattr(lc, f"{src_stream}_ff_error", None)
        if error is None:
            return edges
        error = error.to(device, dtype=torch.float32)

        gate = getattr(lc, f"{src_stream}_gate_mlp", None)
        if gate is not None:
            gate = gate.to(device, dtype=torch.float32)
            if gate.dim() == 2:
                gate = gate.unsqueeze(1)
        else:
            gate = torch.ones_like(error)

        write_layer = min(src_layer + 1, target_node.layer)
        v = vjp_grads.get((write_layer, src_stream))
        if v is None:
            return edges
        v = v.to(device, dtype=torch.float32)

        gated_error = error[0] * gate[0]
        attrs_per_pos = (gated_error * v[0]).sum(dim=-1)

        mask = attrs_per_pos.abs() >= self.cfg.circuit_min_attribution
        valid_pos = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if valid_pos.numel() == 0:
            return edges

        pos_cpu = valid_pos.cpu().numpy()
        attr_cpu = attrs_per_pos[valid_pos].cpu().numpy()

        for idx in range(len(pos_cpu)):
            source_id = NodeId(
                node_type=NodeType.ERROR,
                layer=src_layer,
                stream=src_stream,
                position=int(pos_cpu[idx]),
            )
            edges.append(
                EdgeData.for_error(
                    source=source_id,
                    target=target_node,
                    error_contrib=float(attr_cpu[idx]),
                )
            )

        return edges

    def _compute_boundary_edges_per_position(
        self,
        trace: FluxTrace,
        target_node: NodeId,
        vjp_grads: Dict[Tuple[int, str], Optional[Tensor]],
    ) -> List[EdgeData]:
        edges = []
        device = self.cfg.device

        for stream in ["img", "txt"]:
            boundary_key = (self.cfg.first_layer - 1, stream)
            v = vjp_grads.get(boundary_key)
            if v is None:
                continue

            boundary = trace.boundary_img if stream == "img" else trace.boundary_txt
            if boundary is None:
                continue

            boundary = boundary.to(device, dtype=torch.float32)
            v = v.to(device, dtype=torch.float32)

            attrs_per_pos = (boundary[0] * v[0]).sum(dim=-1)
            mask = attrs_per_pos.abs() >= self.cfg.circuit_min_attribution
            valid_pos = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if valid_pos.numel() == 0:
                continue

            pos_cpu = valid_pos.cpu().numpy()
            attr_cpu = attrs_per_pos[valid_pos].cpu().numpy()

            for idx in range(len(pos_cpu)):
                source_id = NodeId(
                    node_type=NodeType.INPUT,
                    layer=self.cfg.first_layer,
                    stream=stream,
                    position=int(pos_cpu[idx]),
                )
                edges.append(
                    EdgeData.for_residual(
                        source=source_id,
                        target=target_node,
                        residual_contrib=float(attr_cpu[idx]),
                    )
                )

        return edges
