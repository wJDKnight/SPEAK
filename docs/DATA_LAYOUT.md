# Data-package layout

The GitHub repository includes the compact STARmap example under `examples/starmap/` and
three small source-data tables under `examples/source_data/`. The complete paper data are
distributed through Figshare at https://doi.org/10.6084/m9.figshare.33043091.

Download the deposited files and place them under `examples/` using the layout below. An
equivalent external location can be selected with the `LLM_ST_*` environment variables in
`environment/.env.example`.

```text
examples/
├── starmap/
│   ├── BZ5|BZ9|BZ14/{data,celltype,pos,domain}.csv
│   └── representative_genes.txt
├── visium_libd/
│   ├── <12 sample IDs>/
│   │   ├── filtered_feature_bc_matrix.h5
│   │   ├── metadata.tsv
│   │   ├── spatial/tissue_positions_list.csv
│   │   └── celltype_proportions_<sample>.csv
│   └── deconv_result/
├── merfish/
│   ├── MERFISH_25.h5ad ... MERFISH_29.h5ad
│   ├── representative_genes.txt
│   └── gene_sets.tsv
├── copd/230267_Slide2/
│   ├── processed method input and selected-cell IDs
│   ├── zero-shot and fine-tuned annotations
│   └── DE and marker-expression source tables
├── results/
│   ├── row-level annotations
│   ├── benchmark metric tables
│   ├── prototype-sensitivity outputs
│   └── k-means baseline results
├── source_data/
│   ├── noise_experiment.csv
│   ├── runtime_reference.csv
│   └── ground_truth_continuity.csv
└── intermediates/
    ├── model_requests/
    ├── model_responses/
    ├── finetuning/
    ├── local_llm_results/
    ├── visium_libd_deconvolution/
    └── merfish/
```

The Figshare record is the authoritative source for the complete deposited file inventory.
