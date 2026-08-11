# STARmap processed inputs

This directory contains every analysis input used for the STARmap experiments.
The three samples have the same file layout and share the representative-gene
list at the directory root.

## Samples

| Sample directory | Cells | Measured genes | Files |
|---|---:|---:|---|
| `BZ5/` | 1,049 | 166 | `data.csv`, `celltype.csv`, `pos.csv`, `domain.csv` |
| `BZ9/` | 1,053 | 166 | `data.csv`, `celltype.csv`, `pos.csv`, `domain.csv` |
| `BZ14/` | 1,088 | 166 | `data.csv`, `celltype.csv`, `pos.csv`, `domain.csv` |

## File definitions

The following definition applies separately to every file in `BZ5/`, `BZ9/`,
and `BZ14/`.

| Filename | Contents and use |
|---|---|
| `<sample>/data.csv` | Cell-by-gene expression matrix. The first column contains the stable STARmap cell identifier; the remaining 166 columns are measured genes. This is the expression input used to derive neighborhood gene summaries. |
| `<sample>/celltype.csv` | One row per cell with column `c`, the published/reference cell-type label. Row identifiers exactly match `data.csv`. These labels form the neighborhood cell-type counts supplied to the method. |
| `<sample>/pos.csv` | One row per cell with original spatial coordinates `x` and `y`. Row identifiers exactly match the other sample files. These coordinates are used to construct spatial neighbors and calculate CHAOS, PAS, and ASW. |
| `<sample>/domain.csv` | One row per cell with column `z`, the reference spatial-domain label used as ground truth for evaluation. The numeric codes are preserved from the processed STARmap source. |
| `representative_genes.txt` | The exact ordered list of 16 representative genes read by the STARmap zero-shot and fine-tuned workflows. There is one gene symbol per line and no header. |

## Alignment and provenance

Within a sample, all four CSV files contain the same cells in compatible order
and can also be joined by the first-column identifier. The CSV files were
exported from the processed STARmap objects by
`workflows/starmap/extract_starmap.r`; the representative genes are consumed
directly by the STARmap notebooks and scripts.
