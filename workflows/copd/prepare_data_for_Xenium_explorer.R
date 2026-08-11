# extract zeroshot and seuratnihce results from my pipline for visualization in Xenium explorer
library(readr)
slide_number = 2
copd_root <- Sys.getenv("LLM_ST_COPD_ROOT", "examples/copd/230267_Slide2")
sample_dir <- file.path(copd_root, "xenium_explorer")
dir.create(sample_dir, showWarnings = FALSE, recursive = TRUE)
my_results <- read_csv(file.path(copd_root, paste0("230267_Slide", slide_number, "_finetune_gpt4o_refined_k20.csv")))

celltype = my_results[c('orig.ident', 'cell')]
colnames(celltype) = c("cell_id", "group")
celltype$cell_id = gsub(paste0("Slide", slide_number,"_"), "", celltype$cell_id)
write_csv(celltype, file.path(sample_dir, "celltype.csv"))


zeroshot = my_results[c('orig.ident', 'zeroshot_gpt4o_mini_refined')]
colnames(zeroshot) = c("cell_id", "group")
zeroshot$cell_id = gsub(paste0("Slide", slide_number,"_"), "", zeroshot$cell_id)
write_csv(zeroshot, file.path(sample_dir, "custom_group_zeroshot.csv"))

seurat_k = my_results[c('orig.ident', 'k.niches.8')]
colnames(seurat_k) = c("cell_id", "group")
seurat_k$cell_id = gsub(paste0("Slide", slide_number,"_"), "", seurat_k$cell_id)
write_csv(seurat_k, file.path(sample_dir, "custom_group_seurat_k8.csv"))

finetune = my_results[c('orig.ident', 'finetune_gpt4o_mini_refined')]
colnames(finetune) = c("cell_id", "group")
finetune$cell_id = gsub(paste0("Slide", slide_number,"_"), "", finetune$cell_id)
write_csv(finetune, file.path(sample_dir, "custom_group_finetune.csv"))
