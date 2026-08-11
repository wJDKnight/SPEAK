library(tidyverse)
library(Matrix)
source("workflows/starmap/functions.R")


args <- commandArgs(trailingOnly = TRUE)
data_name <- if (length(args) >= 1) args[[1]] else "BZ5"
starmap_root <- Sys.getenv("LLM_ST_STARMAP_ROOT", "examples/starmap")
path_template <- Sys.getenv("LLM_ST_STARMAP_TEMPLATE", "{sample}")
data_path <- file.path(starmap_root, gsub("\\{sample\\}", data_name, path_template))
x_data_name = "normalized_data.csv"

#check whether the row name are saved in csv
rownames = NULL
celltype_data = read.csv(file.path(data_path, "celltype.csv"), row.names = rownames)
if(ncol(celltype_data) == 2) rownames=1

celltype_data = read.csv(file.path(data_path, "celltype.csv"), row.names = rownames)
colnames(celltype_data) = "cell_type"
x_data = read.csv(file.path(data_path, x_data_name), row.names = rownames)
pos_data = read.csv(file.path(data_path, "pos.csv"), row.names = rownames)
domain_data = read.csv(file.path(data_path, "domain.csv"), row.names = rownames)
colnames(domain_data) = "niche_truth"

#plot_radius_degree(pos_data, label_n = 10)

r = 700
adj_matrix = sparse_adjacency(pos_data, threshold = r)
gc()

# Generate one-hot encoded matrix
one_hot_matrix <- model.matrix(~ cell_type - 1, data = celltype_data)
colnames(one_hot_matrix) = gsub("cell_type", "", colnames(one_hot_matrix))
one_hot_matrix <- as(one_hot_matrix, "sparseMatrix")

neighbor_count = adj_matrix %*% one_hot_matrix
rownames(neighbor_count) = rownames(one_hot_matrix)
write.csv(as.matrix(neighbor_count), file.path(data_path, "neighbor_count.csv"), row.names=FALSE)
