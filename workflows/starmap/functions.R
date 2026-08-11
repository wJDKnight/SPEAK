#####function####
# Define a function to create a sparse adjacency matrix
sparse_adjacency <- function(locations, threshold) {
  # Compute pairwise distances using dist()
  dist_matrix <- dist(locations)
  
  # Convert the dist object into a matrix of distances (symmetric)
  dist_matrix <- as.matrix(dist_matrix)
  
  # Apply the threshold to filter out large distances and keep only neighbors
  adj_list <- which(dist_matrix <= threshold & dist_matrix != 0, arr.ind = TRUE)
  
  # Construct the sparse adjacency matrix
  adj_matrix_sparse <- sparseMatrix(i = adj_list[, 1], 
                                    j = adj_list[, 2], 
                                    x = 1, 
                                    dims = dim(dist_matrix))
  
  return(adj_matrix_sparse)
}

plot_radius_degree = function(locations, label_n = 100){
  # label_n is for labelling the radius where you get the number of label_n neighbors
  # 查找每个点在半径r内的邻居
  scale_range = max(locations[,1]) - mean(locations[,1])
  
  r = seq(scale_range*0.01,scale_range*0.2, length.out = 25)
  n_neighbors = sapply(r, function(r){
    neighbors <- frNN(locations, eps = r)
    return(mean(sapply(neighbors$id, length)))
  })
  
  plot_df = data.frame(r = r, n_neighbors = n_neighbors, proportion = n_neighbors/nrow(locations))
  plot_df = plot_df %>% filter(proportion<0.10)
  p = plot_df %>%
    ggplot() +
    aes(x = r, y = n_neighbors, colour = proportion) +
    geom_point(shape = "circle", size = 1.5) +
    scale_color_viridis_c(option = "viridis", direction = 1) +
    labs(title = "The size of neighborhood over different radius") +
    theme_minimal() +
    theme(plot.title = element_text(size = 15L))+
    geom_text(aes(x = r[which.min(abs(n_neighbors - label_n))],
                  y = n_neighbors[which.min(abs(n_neighbors - label_n))], 
                  label=paste0(round(r[which.min(abs(n_neighbors - label_n))],1),",", round(n_neighbors[which.min(abs(n_neighbors - label_n))],1))),
              color = 'red', vjust=-1) 
  return(p)
}
