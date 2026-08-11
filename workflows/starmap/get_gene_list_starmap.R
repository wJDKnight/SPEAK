
library(zellkonverter)
library(Seurat)
library(patchwork)
library(ggplot2)
library(VennDiagram)
library(dplyr)
library(tidyverse)

data_names = c("BZ5", "BZ9", "BZ14")
starmap_root <- Sys.getenv("LLM_ST_STARMAP_ROOT", "examples/starmap")
path_template <- Sys.getenv("LLM_ST_STARMAP_TEMPLATE", "{sample}")
# Create list to store Seurat objects
seurat_list <- list()

for (data_name in data_names) {
    data_path <- file.path(starmap_root, gsub("\\{sample\\}", data_name, path_template))
    counts = read.csv(paste0(data_path, "/data.csv"), row.names = 1)
    pos = read.csv(paste0(data_path, "/pos.csv"), row.names = 1)
    cell_type = read.csv(paste0(data_path, "/celltype.csv"), row.names = 1)
    domain = read.csv(paste0(data_path, "/domain.csv"), row.names = 1)
    obj = CreateSeuratObject(counts = t(counts), project = data_name)
    obj = AddMetaData(obj, pos)
    obj = AddMetaData(obj, cell_type)
    obj = AddMetaData(obj, domain)
    obj$sample = data_name
    seurat_list[[data_name]] = obj
    
}
merged_obj <- merge(seurat_list[[1]], 
                   y = seurat_list[2:length(seurat_list)], 
                   add.cell.ids = data_names,
                   project = "STARmap")

merged_obj = NormalizeData(merged_obj)
merged_obj = FindVariableFeatures(merged_obj, nfeatures = 100)
merged_obj = ScaleData(merged_obj)
merged_obj = RunPCA(merged_obj, features = VariableFeatures(object = merged_obj))
integrated_obj <- IntegrateLayers(
    object = merged_obj,
    method = CCAIntegration,
    orig.reduction = "pca",
    new.reduction = "integrated",
    dims = 1:30,
    verbose = FALSE
)

# Run UMAP on integrated data
integrated_obj <- FindNeighbors(integrated_obj, reduction = "integrated", dims = 1:20)
integrated_obj <- FindClusters(integrated_obj, resolution = 0.3)
integrated_obj <- RunUMAP(integrated_obj, reduction = "integrated", dims = 1:20)

# Plot UMAP after integration
p1 <- DimPlot(integrated_obj, reduction = "umap", group.by = "seurat_clusters")
p2 <- DimPlot(integrated_obj, reduction = "umap", group.by = "sample") 
p1|p2

p3 <- DimPlot(integrated_obj, group.by = "c")
p4 <- DimPlot(integrated_obj, group.by = "z")
p3|p4

# join layers and run find conserved markers
merged_obj = JoinLayers(integrated_obj)
merged_obj = NormalizeData(merged_obj)

n_top = 5
gene_names = c()
Idents(merged_obj) = "c"  # seurat_clusters, c (cell type)
for(cell_type in unique(merged_obj$c)){
  
    markers <- FindConservedMarkers(merged_obj, ident.1 = cell_type, assay = "RNA", 
                                    grouping.var = "sample", verbose = FALSE, only.pos = TRUE)
    # Select columns ending with _avg_log2FC and sum them
    markers$total_log2FC <- markers %>%
      select(ends_with("avg_log2FC")) %>%
      rowSums()

    markers$min_pct.1 = apply(markers[,grepl("pct.1", colnames(markers))], 1, min)
    markers$min_log2FC = apply(markers[,grepl("avg_log2FC", colnames(markers))], 1, min)
    markers$max_p_val_adj = apply(markers[,grepl("p_val_adj", colnames(markers))], 1, max)
    
    # Sort markers by total log2FC in descending order  
    markers <- markers %>%
      arrange(desc(total_log2FC)) %>%
      filter(min_pct.1>0.3) %>%
      filter(min_log2FC>1.5) %>%
      filter(max_p_val_adj<0.05)
    gene_names <- c(gene_names, rownames(markers)[1:n_top])
}
gene_names = unique(gene_names) %>% na.omit()
print(gene_names)

FeaturePlot(merged_obj, rownames(markers)[1:n_top])
gene_names = c("Aqp4", "Gad1", "Gad2", "Sst", "Rbp4", "Cux2", "Synpr", "Nos1", "Cpne5", "Bdnf", "Adcyap1", "Syt6", "Pcp4", "Sla", "Ctgf", "Foxp2")
gene_m = FetchData(merged_obj, vars = gene_names, layer = "data")
plot_df = cbind(merged_obj[[]][c('x','y','c','z')], gene_m)
plot_df = plot_df[grepl("BZ5", rownames(plot_df)),]

ggplot(plot_df, aes(x,y, color = Mylk) )+
  geom_point()+
  theme_minimal()+
  scale_color_viridis_c()

ggplot(plot_df, aes(x,y, color = as.factor(z))) +
  geom_point()+
  theme_minimal()


# violin plot for each gene in each z
long_df <- plot_df %>%
  select(all_of(c("z", gene_names))) %>%
  mutate(z = as.factor(z)) %>%
  pivot_longer(
    cols = all_of(gene_names),
    names_to = "gene",
    values_to = "expression"
  ) %>%
  mutate(gene = factor(gene, levels = gene_names))

p <- ggplot(long_df, aes(x = gene, y = expression, fill = gene)) +
  geom_violin(scale = "width", trim = FALSE, color = "gray30", linewidth = 0.2) +
  facet_grid(rows = vars(z), scales = "fixed") +  # one stacked panel per class z
  labs(x = "Gene", y = "Expression level") +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1)
  )

print(p)

# violin plot for each gene in each c
long_df <- plot_df %>%
  select(all_of(c("c", gene_names))) %>%
  mutate(c = as.factor(c)) %>%
  pivot_longer(
    cols = all_of(gene_names),
    names_to = "gene",
    values_to = "expression"
  ) %>%
  mutate(gene = factor(gene, levels = gene_names))

p <- ggplot(long_df, aes(x = gene, y = expression, fill = gene)) +
  geom_violin(scale = "width", trim = FALSE, color = "gray30", linewidth = 0.2) +
  facet_grid(rows = vars(c), scales = "fixed") +  # one stacked panel per class c
  labs(x = "Gene", y = "Expression level") +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1)
  )

print(p)


# print cell type and its marker genes
# open a file to print the result
file_path <- file.path(starmap_root, "marker_gene_queries.txt")
file_conn = file(file_path, open = "w")

for(cell_type in unique(merged_obj$c)){
  
    markers <- FindConservedMarkers(merged_obj, ident.1 = cell_type, assay = "RNA", 
                                    grouping.var = "sample", verbose = FALSE, only.pos = TRUE)
    # Select columns ending with _avg_log2FC and sum them
    markers$total_log2FC <- markers %>%
      select(ends_with("avg_log2FC")) %>%
      rowSums()

    markers$min_pct.1 = apply(markers[,grepl("pct.1", colnames(markers))], 1, min)
    markers$min_log2FC = apply(markers[,grepl("avg_log2FC", colnames(markers))], 1, min)
    markers$max_p_val_adj = apply(markers[,grepl("p_val_adj", colnames(markers))], 1, max)
    
    # Sort markers by total log2FC in descending order  
    markers <- markers %>%
      arrange(desc(total_log2FC)) %>%
      filter(min_pct.1>0.3) %>%
      filter(min_log2FC>1.5) %>%
      filter(max_p_val_adj<0.05)
    
    selected_genes = rownames(markers)[1:n_top]
    selected_genes = na.omit(selected_genes)
    if(length(selected_genes) > 0){
        writeLines(paste0("Does the ", selected_genes, " is the marker genes of the cell type: ", cell_type, "?"), file_conn) 
        writeLines("--------------------------------", file_conn)
    }
}

close(file_conn)
