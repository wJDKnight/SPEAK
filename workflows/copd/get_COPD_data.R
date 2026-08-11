# get copd xeium data
library(Seurat)
library(tidyverse)
copd_root <- Sys.getenv("LLM_ST_COPD_ROOT", "examples/copd/230267_Slide2")
xenium_rds <- Sys.getenv("LLM_ST_COPD_XENIUM_RDS", file.path(copd_root, "xenium_230267_Slide2.rds"))
all_data <- readRDS(xenium_rds)

# remove doublets
all_data = all_data[,all_data$cell!="doublet"]


sample_id = '230267'
slide_id = "Slide2"
fov_name = 'fov.2'

sample_obj <- subset(all_data, subset = Sample == sample_id) %>%
  subset(subset = Slide == slide_id)

sample_metadata = sample_obj[[]]

pca_df = as.data.frame(sample_obj[['pca']]@cell.embeddings[,1:3]) %>%
  rownames_to_column("orig.ident")

# umap_df = as.data.frame(sample_metadata[['umap']]@cell.embeddings[,1:2]) %>%
#   rownames_to_column("orig.ident")

sample_metadata = sample_metadata %>% 
  left_join(pca_df, by = "orig.ident") 

x_y = as.data.frame(sample_obj[[fov_name]]$centroids@coords)
x_y$orig.ident = sample_obj[[fov_name]]$centroids@cells

sample_metadata = sample_metadata %>% 
  left_join(x_y, by = "orig.ident") 

write_csv(sample_metadata, 
          file.path(copd_root, paste0("sample_", sample_id, "_", slide_id, "_k20.csv")))
