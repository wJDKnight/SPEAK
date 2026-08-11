# seurat default niches analysis
# niches should be sperately done across slides, because the coordiantes across slides may be over lapped
library(Seurat)
library(tidyverse)
copd_root <- Sys.getenv("LLM_ST_COPD_ROOT", "examples/copd/230267_Slide2")
xenium_rds <- Sys.getenv("LLM_ST_COPD_XENIUM_RDS", file.path(copd_root, "xenium_230267_Slide2.rds"))
all_data <- readRDS(xenium_rds)


sample_id = "230267"
slide_id = "Slide2"
fov_name = 'fov.2'

sample_obj <- subset(all_data, subset = Sample == sample_id) %>%
  subset(subset = Slide == slide_id)



sample_obj = sample_obj[,sample_obj$cell!="doublet"]

niche_dataframe = data.frame(row.names = rownames(sample_obj[[]]))

for (k_niche in c(6, 7, 8, 9, 10, 11)) {
  sample_obj <- BuildNicheAssay(object = sample_obj, fov = "fov", group.by = "cell",
                                niches.k = k_niche, neighbors.k = 30)

  # niche.plot <- ImageDimPlot(sample_obj, group.by = "niches", size = 1.5, dark.background = F) + ggtitle(paste0("Niches: ", k_niche)) 
  # print(niche.plot)

  niche_dataframe[paste0("seurat_niche_", k_niche)] = sample_obj$niches
}


output_dir <- copd_root
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
write_csv(niche_dataframe, file.path(output_dir, paste0("niche_dataframe_", slide_id, ".csv")))
