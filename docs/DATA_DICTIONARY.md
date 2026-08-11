# Data dictionary

## Row-level annotations

| Field | Meaning |
|---|---|
| cell_id / spot_id / orig.ident | Stable observation identifier from the processed input |
| data_type | `starmap`, `visium`, `merfish`, or `copd` |
| data_name | Sample identifier |
| x, y | Spatial coordinates in the units supplied by the source dataset |
| ground_truth / niche_truth / layer_guess | Reference domain label used for evaluation |
| zeroshot_* | First-stage zero-shot prediction |
| finetune_* | First-stage fine-tuned prediction |
| *_refined | Spatially refined prediction |
| *_twostage | Second-stage model prediction |
| confident_cells | Indicator used to construct prototype/training subsets |
| model_name / model_type | Model or experimental condition |
| replicate | Independent model run identifier |

Dataset-specific aliases must be documented in this repository rather
than silently renamed.  Coordinates must retain their original unit or explicitly
record that they are array/pixel coordinates.

## Benchmark metrics

| Field | Direction | Definition |
|---|---|---|
| ARI | higher | Adjusted Rand index against the reference labels |
| NMI | higher | Normalized mutual information against the reference labels |
| HOM | higher | Homogeneity score |
| COM | higher | Completeness score |
| CHAOS | lower | Mean within-domain nearest-neighbour distance after coordinate scaling |
| PAS | lower | Fraction whose ten-neighbour majority has a different predicted label |
| ASW | higher | Euclidean spatial average silhouette width |
| time | lower | Runtime; units must be recorded in the figure source file |
| memory | lower | Peak memory; units must be recorded in the deposited metadata |

Missing values are represented by empty fields in source/metric tables and
must not be encoded as zero.
