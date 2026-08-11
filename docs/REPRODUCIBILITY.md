# Reproducibility guide

## Configuration and data setup

The repository includes all source code, configurations, the STARmap BZ5/BZ9/BZ14 example
inputs, and three small source-data tables. Download the complete paper data from
https://doi.org/10.6084/m9.figshare.33043091 and place the deposited directories under
`examples/` as documented in `docs/DATA_LAYOUT.md`. Alternatively, set the `LLM_ST_*`
variables in `environment/.env.example` to an external data location. No author home
directory is required.

## Dataset workflows

### STARmap

1. Read BZ5/BZ9/BZ14 `data.csv`, `celltype.csv`, `pos.csv`, and `domain.csv`
   from `examples/starmap/<sample>/`.
2. Read the exact method gene list from `examples/starmap/representative_genes.txt`.
3. Run the zero-shot/fine-tuned workflows and refinement.
4. After obtaining the deposited results, recompute metrics from `examples/results/`.

### Visium / spatialLIBD

1. Download each sample matrix, metadata, coordinates, and
   `celltype_proportions_<sample>.csv` into `examples/visium_libd/<sample>/`.
2. Run the zero-shot and prototype fine-tuning workflows.
3. Reuse or regenerate the deposited deconvolution artifacts under
   `examples/intermediates/visium_libd_deconvolution/`.

### MERFISH

1. Download MERFISH_25–MERFISH_29 H5AD files into `examples/merfish/`.
2. Use the deposited `representative_genes.txt` and `gene_sets.tsv` files.
3. Reuse or regenerate the integrated R objects under `examples/intermediates/merfish/`.
4. Run zero-shot, fine-tuned, refinement, and noise analyses.

### COPD

1. Download the deposited Slide2 files into `examples/copd/230267_Slide2/`.
2. Use `sample_230267_Slide2_k20.csv` as the method input and
   `selected_cells/selected_cells_id.csv` as the selected fine-tuning cells.
3. Use the deposited zero-shot and final fine-tuned annotations for comparison.
4. Use `230267_Slide2_SMC_de_full.csv` and `Slide2_marker_expression_source.csv`
   to rebuild the DE and spatial-marker panels.

## Model audit and expensive outputs

The exact paper-scope requests, responses, fine-tuning records, deconvolution models,
latent embeddings, and MERFISH R objects are distributed in the Figshare data package
under `examples/intermediates/`. Provider credentials and non-paper experiments are not
included.

## Offline verification

After installing the package, verify the compact GitHub release with:

```bash
pytest
speak-validate-release --strict
speak-smoke-test
```

To validate a downloaded full data package as well, run:

```bash
speak-validate-release --strict --data-root examples
```

API-based inference is not part of the offline smoke test because it requires provider
credentials, can incur charges, and may not be deterministic. Deposited request and response
records allow the reported runs to be audited without repeating API calls.
