library(Seurat)
library(readr)
library(dplyr)
library(tibble)

copd_root <- Sys.getenv("LLM_ST_COPD_ROOT", "examples/copd/230267_Slide2")
source_rds <- Sys.getenv("LLM_ST_COPD_SOURCE_RDS")
if (source_rds == "" || !file.exists(source_rds)) {
  stop("Set LLM_ST_COPD_SOURCE_RDS to the public source Seurat RDS")
}

xenium_obj <- readRDS(source_rds)
xenium_obj <- subset(xenium_obj, subset = Sample == "230267" & Slide == "Slide2")
xenium_obj <- xenium_obj[, xenium_obj$cell != "doublet"]

annotation <- read_csv(
  file.path(copd_root, "230267_Slide2_finetune_gpt4o_refined_k20.csv"),
  show_col_types = FALSE
) %>%
  select(orig.ident, finetune_gpt4o_mini_refined)
annotation <- column_to_rownames(annotation, "orig.ident")
xenium_obj <- AddMetaData(xenium_obj, annotation)

smc_obj <- subset(xenium_obj, subset = cell == "SMC")
de_markers <- FindMarkers(
  smc_obj,
  ident.1 = "Airway Inflamed",
  ident.2 = "Large Vessel Inflamed",
  group.by = "finetune_gpt4o_mini_refined"
) %>%
  rownames_to_column("gene") %>%
  mutate(
    significant = p_val_adj < 0.05 & abs(avg_log2FC) > 2,
    group = if_else(avg_log2FC > 0, "Airway Inflamed", "Large Vessel Inflamed")
  )
write_csv(de_markers, file.path(copd_root, "230267_Slide2_SMC_de_full.csv"))

selected <- read_csv(
  file.path(copd_root, "Slide2_focus_cells_stats.csv"),
  show_col_types = FALSE
)
selected_ids <- paste0("Slide2_", selected$`Cell ID`)
selected_ids <- intersect(selected_ids, colnames(xenium_obj))
genes <- intersect(c("FOXJ1", "TP63", "MUC5AC", "VWF", "ACTA2"), rownames(xenium_obj))

expression <- FetchData(xenium_obj, vars = genes, cells = selected_ids) %>%
  rownames_to_column("orig.ident")
coordinates <- read_csv(
  file.path(copd_root, "sample_230267_Slide2_k20.csv"),
  show_col_types = FALSE
) %>%
  select(orig.ident, x, y)
marker_source <- expression %>%
  left_join(coordinates, by = "orig.ident") %>%
  left_join(
    annotation %>% rownames_to_column("orig.ident"),
    by = "orig.ident"
  )
write_csv(marker_source, file.path(copd_root, "Slide2_marker_expression_source.csv"))
