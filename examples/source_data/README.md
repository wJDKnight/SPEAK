# Machine-readable source data

These three small CSV files externalize numeric values that were formerly embedded in
analysis notebooks. They are retained in GitHub for transparent inspection; the complete
row-level results and additional figure-supporting files are distributed through Figshare.

| File | Dimensions | Contents |
|---|---:|---|
| `ground_truth_continuity.csv` | 9 rows × 5 columns | Ground-truth continuity references for STARmap, Visium and MERFISH, with `data_type`, metric (`CHAOS`, `PAS` or `ASW`), mean, standard deviation and provenance. |
| `noise_experiment.csv` | 9 rows × 4 columns | Noise-sensitivity values at noise fractions 0.0–0.8, including the mean number of changed top-ten genes, NMI and performance loss. |
| `runtime_reference.csv` | 3 rows × 3 columns | Reported LLM runtime references for STARmap, Visium and MERFISH; runtime units are seconds. |

Column definitions shared with deposited result files are documented in
`docs/DATA_DICTIONARY.md`.
