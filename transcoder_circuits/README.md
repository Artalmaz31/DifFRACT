# `transcoder_circuits`

Tools for reverse-engineering and analyzing circuits with the trained transcoders — the machinery
used by the walkthrough and the case studies.

| File | What it holds |
|---|---|
| `replacement_model.py` | The Local Replacement Model (frozen joint attention, frozen-denominator LayerNorm, transcoder + error substitution) and trace capture. |
| `attribution_graph.py` | The graph / node / edge data structures and position aggregation. |
| `edges.py` | The VJP and edge computation (feature, error and input attributions). |
| `influence.py` | The indirect-influence linear algebra, `B = (I − A_norm)⁻¹ − I`. |
| `tracing.py` | `CircuitTracer` — iterative budgeted expansion and the effective bias terms. |
| `pruning.py` | `GraphPruner` — two-step, per-stream cumulative pruning. |
| `pipeline.py` | `FluxLRMPipeline`, the end-to-end driver that traces, aggregates, prunes and validates a circuit and writes it to JSON. |
| `validation.py` | The conservation-invariant and perturbation-faithfulness checks. |
| `circuit_analysis.py` | An umbrella module re-exporting the graph-construction API. |
| `feature_dashboards.py` | The two-pass corpus scan and feature dashboards, the automated `FeatureScanner`, and `build_feature_tooltips` for embedding dashboards into graph tooltips. |
| `circuit_visualization.py` | Interactive `pyvis` rendering of saved attribution graphs. |
| `interventions.py` | Circuit-guided steering (per-feature, per-step, per-stream). |
