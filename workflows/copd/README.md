# COPD Xenium workflow

Scope is strictly `230267_Slide2`.

All required files are deposited in Figshare. After download, place them below
`examples/copd/230267_Slide2/`:

- `sample_230267_Slide2_k20.csv`
- `230267_Slide2_zeroshot_gpt4o_refined_k20.csv`
- `selected_cells/selected_cells_id.csv`
- `230267_Slide2_finetune_gpt4o_refined_k20.csv`
- `230267_Slide2_SMC_de_full.csv`
- `Slide2_marker_expression_source.csv`

`plot_DE_genes_Xenium.R` and `sankey_plot.R` rebuild the COPD panels directly
from these files.
