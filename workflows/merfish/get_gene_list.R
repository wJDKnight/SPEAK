library(zellkonverter)
library(Seurat)
library(patchwork)
library(ggplot2)
library(VennDiagram)
library(dplyr)
data_names <- c("MERFISH_25", "MERFISH_26", "MERFISH_27", "MERFISH_28", "MERFISH_29")
merfish_root <- Sys.getenv("LLM_ST_MERFISH_ROOT", "examples/merfish")
intermediate_root <- Sys.getenv("LLM_ST_INTERMEDIATE_ROOT", "examples/intermediates/merfish")
dir.create(intermediate_root, showWarnings = FALSE, recursive = TRUE)

# Create list to store Seurat objects
seurat_list <- list()

# Read each h5ad file and convert to Seurat
for (i in seq_along(data_names)) {
    ad <- readH5AD(file.path(merfish_root, paste0(data_names[i], ".h5ad")))
    seurat_obj <- as.Seurat(ad, counts = "X", data = NULL)
    # Add sample ID to metadata
    seurat_obj$sample <- data_names[i]
    seurat_list[[i]] <- seurat_obj
}

# Merge all Seurat objects
# The first object is used as base, then merge others into it
merged_obj <- merge(seurat_list[[1]], 
                   y = seurat_list[2:length(seurat_list)], 
                   add.cell.ids = data_names,
                   project = "MERFISH")

merged_obj$cell_class <- recode(merged_obj$cell_class,
                                "Endothelial 1" = "Endothelial",
                                "Endothelial 2" = "Endothelial",
                                "Endothelial 3" = "Endothelial",
                                "OD Mature 1" = "OD Mature",
                                "OD Mature 2" = "OD Mature",
                                "OD Mature 3" = "OD Mature",
                                "OD Mature 4" = "OD Mature",
                                "OD Immature 1" = "OD Immature",
                                "OD Immature 2" = "OD Immature")
unique(merged_obj$cell_class)
# "Endothelial" "Inhibitory" "Astrocyte" "Excitatory" "OD Mature" "OD Immature" "Microglia" "Pericytes" "Ependymal"


#split layers
merged_obj[["originalexp"]] <- split(merged_obj[["originalexp"]], f = merged_obj$sample)

# Standard preprocessing
merged_obj <- NormalizeData(merged_obj)
merged_obj <- FindVariableFeatures(merged_obj)
merged_obj <- ScaleData(merged_obj)
merged_obj <- RunPCA(merged_obj)

# Plot UMAP before integration
merged_obj <- RunUMAP(merged_obj, dims = 1:30)
p1 <- DimPlot(merged_obj, group.by = "sample") + 
      ggtitle("Before Integration (sample)")
p3 <- DimPlot(merged_obj, group.by = "cell_class") + 
  scale_color_manual(values = c("Endothelial" = "#1f77b4", "Inhibitory" = "#ff7f0e", "Astrocyte" = "#2ca02c", "Excitatory" = "#d62728", "OD Mature" = "#9467bd", "OD Immature" = "#8c564b", "Microglia" = "#e377c2", "Pericytes" = "#ff7f7f", "Ependymal" = "#bcbd22")) +
  ggtitle("Before Integration (cell type)")

# ---- Perform integration using standard workflow ----
integrated_obj <- IntegrateLayers(
    object = merged_obj,
    method = CCAIntegration,
    orig.reduction = "pca",
    new.reduction = "integrated",
    dims = 1:30,
    verbose = FALSE
)

# re-join layers after integration
integrated_obj[["originalexp"]] <- JoinLayers(integrated_obj[["originalexp"]])

# Run UMAP on integrated data
integrated_obj <- FindNeighbors(integrated_obj, reduction = "integrated", dims = 1:30)
integrated_obj <- FindClusters(integrated_obj, resolution = 0.1)
integrated_obj <- RunUMAP(integrated_obj, reduction = "integrated", dims = 1:30)

# Plot UMAP after integration
p2 <- DimPlot(integrated_obj, group.by = "sample") + 
      ggtitle("After Integration")
p4 <- DimPlot(integrated_obj, group.by = "cell_class") + 
  scale_color_manual(values = c("Endothelial" = "#1f77b4", "Inhibitory" = "#ff7f0e", "Astrocyte" = "#2ca02c", "Excitatory" = "#d62728", "OD Mature" = "#9467bd", "OD Immature" = "#8c564b", "Microglia" = "#e377c2", "Pericytes" = "#ff7f7f", "Ependymal" = "#bcbd22")) + 
  ggtitle("After Integration (cell type)")

p5 <- DimPlot(integrated_obj, group.by = "seurat_clusters") 

# Show plots side by side
p1 + p2

p3 + p4 + p5

# Save the integrated object
saveRDS(merged_obj, file = file.path(intermediate_root, "merged_MERFISH.rda"))
saveRDS(integrated_obj, file = file.path(intermediate_root, "integrated_MERFISH.rda"))
integrated_obj = readRDS(file.path(intermediate_root, "integrated_MERFISH.rda"))







# ---- Identify conserved niche markers ----
Idents(integrated_obj) <- "ground_truth"
gene_names_niche = c()
n_top = 5
for (niche in unique(integrated_obj$ground_truth)) {
    markers <- FindConservedMarkers(integrated_obj, ident.1 = niche, assay = "originalexp", 
                                    grouping.var = "sample", verbose = FALSE)
    
    # Select columns ending with _avg_log2FC and sum them
    markers$total_log2FC <- markers %>%
        select(ends_with("_avg_log2FC")) %>%
        rowSums()
    
    # Sort markers by total log2FC in descending order  
    markers <- markers %>%
        arrange(desc(total_log2FC))
    
    gene_names_niche <- c(gene_names_niche, rownames(markers)[1:n_top])
}
gene_names_niche = unique(gene_names_niche)


# ---- Identify conserved cell type markers ----
Idents(integrated_obj) <- "cell_class"
gene_names_ct = c()
n_top = 5
# get top markers for each cell type
for (cell_type in unique(integrated_obj$cell_class)) {
    markers <- FindConservedMarkers(integrated_obj, ident.1 = cell_type, assay = "originalexp", 
                                    grouping.var = "sample", verbose = FALSE)
    # Select columns ending with _avg_log2FC and sum them
    markers$total_log2FC <- markers %>%
      select(ends_with("_avg_log2FC")) %>%
      rowSums()
    
    # Sort markers by total log2FC in descending order  
    markers <- markers %>%
      arrange(desc(total_log2FC))
    gene_names_ct <- c(gene_names_ct, rownames(markers)[1:n_top])
}
gene_names_ct = unique(gene_names_ct)


# ---- Identify conserved clusters markers ----
Idents(integrated_obj) <- "seurat_clusters"
gene_names_clusters = c()
n_top = 5
for (cluster in unique(integrated_obj$seurat_clusters)) {
    markers <- FindConservedMarkers(integrated_obj, ident.1 = cluster, assay = "originalexp", 
                                    grouping.var = "sample", verbose = FALSE)
    # Select columns ending with _avg_log2FC and sum them
    markers$total_log2FC <- markers %>%
      select(ends_with("_avg_log2FC")) %>%
      rowSums()
    
    # Sort markers by total log2FC in descending order  
    markers <- markers %>%
      arrange(desc(total_log2FC))
    gene_names_clusters <- c(gene_names_clusters, rownames(markers)[1:n_top])
}
gene_names_clusters = unique(gene_names_clusters)

# Create a Venn diagram of conserved markers
venn.plot <- venn.diagram(
  x = list(
    Cell_Type_Markers = gene_names_ct,
    Cluster_Markers = gene_names_clusters
  ),
  category.names = c("Cell Type Markers", "Cluster Markers"),
  filename = NULL,
  imagetype = "png",
  height = 3000,
  width = 3000,
  resolution = 300,
  compression = "lzw",
  lwd = 2,
  col = c("red", "blue"),
  fill = c(alpha("red", 0.5), alpha("blue", 0.5)),
  cex = 2,
  fontface = "bold",
  fontfamily = "sans",
  cat.cex = 2,
  cat.fontface = "bold",
  cat.default.pos = "outer",
  cat.pos = c(-20, 20),
  cat.dist = c(0.05, 0.05),
  cat.fontfamily = "sans",

)

# Plot the Venn diagram
grid.draw(venn.plot)



# ---- Identify highly variable genes for each cell type ----
Idents(integrated_obj) <- "cell_class"
hvgs_cell_class = list()
n_top = 5

for (cell_type in unique(integrated_obj$cell_class)) {
    # Subset the integrated object for the current cell type
    cell_type_obj <- subset(integrated_obj, idents = cell_type)
    
    # Find highly variable genes within the cell type
    cell_type_obj <- FindVariableFeatures(cell_type_obj, selection.method = "vst", nfeatures = n_top)
    
    # Store the top highly variable genes
    hvgs_cell_class[[cell_type]] <- head(VariableFeatures(cell_type_obj), n_top)
}
# Print the highly variable genes for each cell type
hvgs_ct = unique(unlist(hvgs_cell_class))


# ---- Identify highly variable genes for each cluster ----
Idents(integrated_obj) <- "seurat_clusters"
hvgs_clusters = list()
n_top = 5
for (cluster in unique(integrated_obj$seurat_clusters)) {
    cluster_obj <- subset(integrated_obj, idents = cluster)
    cluster_obj <- FindVariableFeatures(cluster_obj, selection.method = "vst", nfeatures = n_top)
    hvgs_clusters[[cluster]] <- head(VariableFeatures(cluster_obj), n_top)
}
hvgs_clusters = unique(unlist(hvgs_clusters))

venn.plot <- venn.diagram(
  x = list(
    Cell_Type_HVGs = hvgs_ct,
    Cluster_HVGs = hvgs_clusters
  ),
  category.names = c("Cell Type HVGs", "Cluster HVGs"),
  filename = NULL,
  imagetype = "png",
  height = 3000,
  width = 3000,
  resolution = 300,
  compression = "lzw",
  lwd = 2,
  col = c("red", "blue"),
  fill = c(alpha("red", 0.5), alpha("blue", 0.5)),
  cex = 2,
  fontface = "bold",
  fontfamily = "sans",
  cat.cex = 2,
  cat.fontface = "bold",
  cat.default.pos = "outer",
  cat.pos = c(-20, 20),
  cat.dist = c(0.05, 0.05),
  cat.fontfamily = "sans",

)

# Plot the Venn diagram
grid.draw(venn.plot)

# Create a list of the gene sets for the Venn diagram
gene_sets <- list(
  "Cell Type HVGs" = hvgs_ct,
  "Cluster HVGs" = hvgs_clusters,
  "Cell Type Genes" = gene_names_ct,
  "Cluster Genes" = gene_names_clusters,
  "Niche Genes" = gene_names_niche
)



# save gene_sets to a csv file
# Convert list to data frame and write to text file
gene_sets_df <- data.frame(
  set_name = names(gene_sets),
  genes = sapply(gene_sets, paste, collapse = ",")
)
write.table(gene_sets_df, file = file.path(merfish_root, "gene_sets.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
writeLines(gene_names_clusters, file.path(merfish_root, "representative_genes.txt"))


# ---- count the overlap between gene_names_niche and other gene lists ----
length(intersect(gene_names_niche, gene_names_ct))/length(gene_names_ct)
length(intersect(gene_names_niche, gene_names_clusters))/length(gene_names_clusters)
length(intersect(gene_names_niche, hvgs_ct))/length(hvgs_ct)
length(intersect(gene_names_niche, hvgs_clusters))/length(hvgs_clusters)
