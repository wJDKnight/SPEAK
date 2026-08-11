# Examples distributed with SPEAK

This compact GitHub release contains a runnable STARmap example and three small
machine-readable source-data tables. The complete paper data package is available in
Figshare at https://doi.org/10.6084/m9.figshare.33043091.

## Files included in GitHub

| Directory | Contents | Purpose |
|---|---|---|
| `starmap/` | BZ5, BZ9 and BZ14 expression, cell-type, coordinate and reference-domain files, plus the representative-gene list | Runnable examples and the offline smoke test |
| `source_data/` | `ground_truth_continuity.csv`, `noise_experiment.csv` and `runtime_reference.csv` | Small source tables supporting reported summaries |

## Files distributed through Figshare

The deposited package additionally contains the complete Visium/spatialLIBD, MERFISH and
COPD inputs; row-level annotations and benchmark results; model requests and responses;
fine-tuning records; local-LLM outputs; deconvolution artifacts; and integrated MERFISH
objects. Place these directories under `examples/` following `docs/DATA_LAYOUT.md`, or use
the `LLM_ST_*` environment variables to select another data root.

## Observation identifiers and missing values

STARmap cell identifiers, Visium barcodes, MERFISH observation indices and COPD Xenium
cell identifiers are retained from their processed source data so inputs and outputs can be
joined without heuristic matching. Empty CSV fields represent missing values unless a
dataset-specific description states otherwise; numeric zero is a measured or computed value
and must not be interpreted as missing.

## Scope and reuse

The paper scope is STARmap BZ5/BZ9/BZ14, Visium/spatialLIBD 151507–151510 and
151669–151676, MERFISH_25–MERFISH_29, and COPD 230267_Slide2. Original public datasets
remain subject to their source licences and should be cited using the identifiers reported in
the manuscript. Generated annotations, metrics and source tables should be cited through the
Figshare record.
