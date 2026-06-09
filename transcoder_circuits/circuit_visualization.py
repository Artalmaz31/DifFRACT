"""Render saved attribution-graph JSONs as interactive pyvis HTML."""

import json
import math
from collections import defaultdict
from typing import Optional

from pyvis.network import Network

# Node palette and edge colors
C_IMG = "#4C72B0"
C_TXT = "#DD8452"
C_ERR = "#d62728"
C_OUT = "#e6a817"
C_INP = "#8B5CF6"
C_POS = "#2563eb"
C_NEG = "#dc2626"


def load_graph(path: str) -> dict:
    """Load attribution-graph JSON into an adjacency-indexed dict."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes_by_id = {n["id"]: n for n in data["nodes"]}

    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    edges = []
    for e in data["edges"]:
        s, t, a = e["src"], e["tgt"], e["attribution"]
        edges.append((s, t, a))
        outgoing[s].append((t, a))
        incoming[t].append((s, a))

    for adj in (outgoing, incoming):
        for k in adj:
            adj[k].sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        "path": path,
        "config": data["config"],
        "target": data["target"],
        "stats": data.get("stats", {}),
        "metadata": data.get("metadata", {}),
        "nodes_by_id": nodes_by_id,
        "edges": edges,
        "outgoing": dict(outgoing),
        "incoming": dict(incoming),
    }


def _node_style(node: dict) -> tuple:
    ntype = node.get("type", "feature")
    stream = node.get("stream", "img")
    if ntype in ("input", "residual"):
        return C_INP, "dot", f"IN {stream}"
    if ntype == "error":
        return C_ERR, "diamond", "E"
    if ntype == "feature":
        color = C_IMG if stream == "img" else C_TXT
        return color, "dot", f"f{node.get('feat_idx', '')}"
    return "#cccccc", "square", str(node.get("feat_idx", ""))


def _node_tooltip(nid, node, graph_data, extra=None) -> str:
    nodes_by_id = graph_data["nodes_by_id"]
    incoming = graph_data["incoming"].get(nid, [])
    outgoing = graph_data["outgoing"].get(nid, [])
    layer, stream = node.get("layer", ""), node.get("stream", "")
    ntype, feat = node.get("type", "feature"), node.get("feat_idx", "")

    lines = [
        f"L{layer} {stream} {ntype} {feat}",
        f"in {len(incoming)} / out {len(outgoing)}",
    ]
    if extra:
        lines.append(extra)
    for label, edges in (("in", incoming), ("out", outgoing)):
        for other_id, a in edges[:5]:
            o = nodes_by_id.get(other_id, {})
            lines.append(
                f"  {label}: L{o.get('layer','')} {o.get('stream','')} "
                f"f{o.get('feat_idx','')}  {a:+.4f}"
            )
    return "\n".join(lines)


def _node_tooltip_html(nid, node, graph_data, direct, dashboard=None) -> str:
    """HTML tooltip: identity, direct attribution, optional activation dashboard,
    then top incident edges. Used when a (dashboard) tooltip_cache is given."""
    nodes_by_id = graph_data["nodes_by_id"]
    incoming = graph_data["incoming"].get(nid, [])
    outgoing = graph_data["outgoing"].get(nid, [])
    layer, stream = node.get("layer", ""), node.get("stream", "")
    ntype, feat = node.get("type", "feature"), node.get("feat_idx", "")
    d = direct.get(nid, 0.0)

    tp = [
        "<div style='max-width:380px;font-family:sans-serif;font-size:12px;'>",
        f"<b>L{layer} {stream} {ntype} {feat}</b><br>",
        f"Direct attr: <b style='color:{C_POS if d > 0 else C_NEG}'>{d:+.4f}</b><br>",
        f"In: {len(incoming)}, Out: {len(outgoing)}<br>",
    ]
    if dashboard:
        tp.append(f"<hr style='margin:4px 0;'>{dashboard}")
    for label, edges in (("Top in", incoming), ("Top out", outgoing)):
        if not edges:
            continue
        tp.append(f"<hr style='margin:4px 0;'><b>{label}:</b><br>")
        for other_id, a in edges[:5]:
            o = nodes_by_id.get(other_id, {})
            tp.append(
                f"L{o.get('layer','')} {o.get('stream','')} f{o.get('feat_idx','')} "
                f"<span style='color:{C_POS if a > 0 else C_NEG}'>{a:+.4f}</span><br>"
            )
    tp.append("</div>")
    return "".join(tp)


# vis.js renders a string node 'title' as plain text, so HTML
# dashboards need to be promoted to DOM nodes after the page loads
_TOOLTIP_PATCH = """
<style>
.vis-tooltip {
  position: absolute !important; padding: 0 !important; background: white !important;
  border: 1px solid #bbb !important; border-radius: 8px !important;
  box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important; z-index: 10000 !important;
  font-family: sans-serif !important; pointer-events: none !important;
}
</style>
<script type="text/javascript">
window.addEventListener('load', function () {
  setTimeout(function () {
    var nodesDS = window.nodes, netObj = window.network;
    if (!nodesDS || !netObj) { return; }
    // Promote HTML-string titles to DOM elements so they render as markup.
    var updated = nodesDS.get().map(function (n) {
      if (n.title && typeof n.title === 'string' && n.title.indexOf('<') !== -1) {
        var c = document.createElement('div');
        c.style.padding = '10px';
        c.innerHTML = n.title;
        n.title = c;
      }
      return n;
    });
    nodesDS.update(updated);
    // Click a node to pin its dashboard; click it again (or the background) to close.
    var panel = document.createElement('div');
    panel.style.cssText = 'position:fixed;top:20px;left:20px;z-index:10001;background:white;' +
      'border:2px solid #aaa;border-radius:10px;padding:15px;display:none;max-width:380px;' +
      'max-height:85vh;overflow-y:auto;box-shadow:0 10px 30px rgba(0,0,0,0.3);font-family:sans-serif;';
    document.body.appendChild(panel);
    var pinned = null;
    netObj.on('click', function (params) {
      if (params.nodes.length > 0) {
        var nid = params.nodes[0];
        if (pinned === nid) { panel.style.display = 'none'; pinned = null; return; }
        var nd = nodesDS.get(nid);
        if (nd && nd.title) {
          pinned = nid;
          panel.innerHTML = '<div style="font-size:10px;color:#999;border-bottom:1px solid #eee;' +
            'margin-bottom:8px;padding-bottom:4px;">pinned &mdash; click node again to close</div>' +
            (nd.title.innerHTML || nd.title);
          panel.style.display = 'block';
        }
      } else { panel.style.display = 'none'; pinned = null; }
    });
  }, 500);
});
</script>
</body>"""


def _apply_tooltip_patch(output_file: str) -> None:
    """Inject the HTML-tooltip render/pin patch into a saved pyvis HTML file."""
    with open(output_file, "r", encoding="utf-8") as f:
        html = f.read()
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html.replace("</body>", _TOOLTIP_PATCH))


def build_interactive_graph(
    graph_data: dict,
    tooltip_cache: Optional[dict] = None,
    idx: int = 0,
    top_edges: Optional[int] = None,
    output_file: str = "circuit_graph.html",
) -> str:
    """Render graph_data to a standalone HTML file."""
    tooltip_cache = tooltip_cache or {}
    html_mode = bool(tooltip_cache)
    nodes_by_id = graph_data["nodes_by_id"]
    all_edges = graph_data["edges"]
    target_id = graph_data["target"]["id"]
    target_layers = graph_data["config"].get("target_layers", list(range(9, 16)))
    max_layer = max(target_layers)

    net = Network(
        notebook=False,
        height="800px",
        width="100%",
        directed=True,
        cdn_resources="in_line",
    )
    net.set_options(
        """
    {
      "physics": {"enabled": false},
      "interaction": {"hover": true, "tooltipDelay": 100, "navigationButtons": true, "keyboard": true},
      "edges": {
        "smooth": {"type": "cubicBezier", "forceDirection": "vertical", "roundness": 0.4},
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}
      },
      "nodes": {"font": {"size": 10, "face": "monospace"}}
    }
    """
    )

    # Direct attribution to the target
    direct = defaultdict(float)
    for src_id, attr in graph_data["incoming"].get(target_id, []):
        direct[src_id] += attr

    # Group feature/error nodes by layer
    grouped = defaultdict(lambda: {"img": [], "txt": [], "other": []})
    input_nodes = []
    for nid, node in nodes_by_id.items():
        if nid == target_id:
            continue
        if node.get("type") in ("input", "residual"):
            input_nodes.append((nid, node))
            continue
        layer = node.get("layer", target_layers[0])
        stream = node.get("stream", "img")
        grouped[layer][stream if stream in ("img", "txt") else "other"].append(
            (nid, node)
        )

    Y_SPACING, X_SPACING, CENTER_GAP = 200, 80, 120

    target_title = tooltip_cache.get(
        target_id, "<b>Target</b>" if html_mode else "target"
    )
    net.add_node(
        target_id,
        label="TARGET",
        title=target_title,
        color=C_OUT,
        size=20,
        shape="star",
        x=0,
        y=0,
        borderWidth=2,
    )

    def place(nid, node, x, y):
        color, shape, label = _node_style(node)
        size = max(8, min(40, 8 + math.log1p(abs(direct.get(nid, 0.0))) * 8))
        if html_mode:
            title = _node_tooltip_html(
                nid, node, graph_data, direct, tooltip_cache.get(nid)
            )
        else:
            title = _node_tooltip(nid, node, graph_data, tooltip_cache.get(nid))
        net.add_node(
            nid,
            label=label,
            title=title,
            color=color,
            size=size,
            shape=shape,
            x=x,
            y=y,
        )

    for layer, streams in grouped.items():
        y = (max_layer - layer) * Y_SPACING + Y_SPACING
        for stream, side in (("img", -1), ("txt", 1)):
            ranked = sorted(
                streams[stream],
                key=lambda nn: abs(direct.get(nn[0], 0.0)),
                reverse=True,
            )
            for i, (nid, node) in enumerate(ranked):
                place(nid, node, side * (CENTER_GAP + i * X_SPACING), y)
        for i, (nid, node) in enumerate(
            sorted(
                streams["other"],
                key=lambda nn: abs(direct.get(nn[0], 0.0)),
                reverse=True,
            )
        ):
            place(nid, node, ((i + 1) // 2 * (1 if i % 2 == 0 else -1)) * X_SPACING, y)

    if input_nodes:
        y = (max_layer + 1) * Y_SPACING + Y_SPACING
        ranked = sorted(
            input_nodes, key=lambda nn: abs(direct.get(nn[0], 0.0)), reverse=True
        )
        for i, (nid, node) in enumerate(ranked):
            side = -1 if node.get("stream", "img") == "img" else 1
            place(nid, node, side * (CENTER_GAP + i * X_SPACING), y)

    sorted_edges = sorted(all_edges, key=lambda e: abs(e[2]), reverse=True)
    display_edges = sorted_edges if top_edges is None else sorted_edges[:top_edges]
    max_attr = max((abs(e[2]) for e in display_edges), default=1.0)
    for s, t, a in display_edges:
        if s not in nodes_by_id or t not in nodes_by_id:
            continue
        color = C_POS if a > 0 else C_NEG
        net.add_edge(
            s,
            t,
            title=f"{a:+.4f}",
            color=f"{color}cc",
            width=max(0.3, abs(a) / max_attr * 5.0),
        )

    net.save_graph(output_file)
    if html_mode:
        _apply_tooltip_patch(output_file)
    print(
        f"Graph[{idx}]: {len(nodes_by_id)} nodes, "
        f"{len(display_edges)}/{len(all_edges)} edges -> {output_file}"
    )
    return output_file


def print_graph_stats(graph_data: dict, idx: int = 0) -> None:
    """Print node/edge counts, direct-attribution split, and conservation error."""
    nodes_by_id = graph_data["nodes_by_id"]
    edges = graph_data["edges"]
    target = graph_data["target"]
    config = graph_data["config"]
    stats = graph_data.get("stats", {})
    target_id = target["id"]

    def is_type(n, *types):
        return n.get("type") in types

    n_feat_img = sum(
        1
        for n in nodes_by_id.values()
        if is_type(n, "feature") and n.get("stream") == "img"
    )
    n_feat_txt = sum(
        1
        for n in nodes_by_id.values()
        if is_type(n, "feature") and n.get("stream") == "txt"
    )
    n_err = sum(1 for n in nodes_by_id.values() if is_type(n, "error"))
    n_inp = sum(1 for n in nodes_by_id.values() if is_type(n, "input", "residual"))

    # Direct attribution to the target
    feat_img = feat_txt = err = inp = 0.0
    for src_id, a in graph_data["incoming"].get(target_id, []):
        node = nodes_by_id.get(src_id, {})
        if is_type(node, "error"):
            err += a
        elif is_type(node, "input", "residual"):
            inp += a
        elif node.get("stream") == "img":
            feat_img += a
        else:
            feat_txt += a
    total = feat_img + feat_txt + err + inp

    # Fraction of feature->feature edges that cross between streams
    total_feat_edges = cross_modal = 0
    for s, t, _ in edges:
        sn, tn = nodes_by_id.get(s, {}), nodes_by_id.get(t, {})
        if sn.get("type") != "feature":
            continue
        if tn.get("type") != "feature" and t != target_id:
            continue
        s_stream = sn.get("stream", "")
        t_stream = target.get("stream", "") if t == target_id else tn.get("stream", "")
        if s_stream and t_stream:
            total_feat_edges += 1
            cross_modal += int(s_stream != t_stream)
    cross_frac = cross_modal / max(total_feat_edges, 1)

    # Conservation: attributions to target should reconstruct h_pre - bias
    attr_sum = stats.get("attribution_sum")
    rel_err = "N/A"
    if attr_sum is not None and target.get("preactivation") is not None:
        expected = target["preactivation"] - target.get("encoder_bias", 0)
        if abs(expected) > 1e-6:
            rel_err = f"{abs(expected - attr_sum) / abs(expected):.2%}"

    print(f"\n{'=' * 70}")
    print(
        f"Graph [{idx}]: L{target['layer']} {target['stream']} f{target['feat_idx']}  "
        f"| {config['prompt'][:60]}"
    )
    print(
        f"  step={config.get('step', '?')}  seed={config.get('seed', '?')}  rel_err={rel_err}"
    )
    print(
        f"  Nodes: {n_feat_img} feat-img, {n_feat_txt} feat-txt, "
        f"{n_err} error, {n_inp} input  ({len(nodes_by_id)} total)"
    )
    print(
        f"  Edges: {len(edges)} total, {total_feat_edges} feat, "
        f"{cross_modal} cross-modal ({cross_frac:.1%})"
    )
    if abs(total) > 1e-8:
        print("  Direct attribution to target:")
        print(
            f"    feat-img: {feat_img:+.4f} ({feat_img/total:.1%})"
            f"  feat-txt: {feat_txt:+.4f} ({feat_txt/total:.1%})"
            f"  error: {err:+.4f} ({err/total:.1%})"
            f"  input: {inp:+.4f} ({inp/total:.1%})"
        )
