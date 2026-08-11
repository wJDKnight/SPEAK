# transfer BASS results from rds to csv
library(Seurat)
library(tidyverse)
library(aricode)
#load data
file_path <- Sys.getenv("LLM_ST_BASS_ROOT", "data/processed_inputs/baselines/BASS-Analysis-master")
load(file.path(file_path, "data/starmap_mpfc.RData"))

# load zlabels
zlabels_K3 = read_rds(file.path(file_path, "BASS_Results/starmap_zlabels_K3.rds"))
zlabels_K4 = read_rds(file.path(file_path, "BASS_Results/starmap_zlabels_K4.rds"))
zlabels_K5 = read_rds(file.path(file_path, "BASS_Results/starmap_zlabels_K5.rds"))

starmap_info$`20180417_BZ5_control`$K3 = zlabels_K3[[1]]
starmap_info$`20180417_BZ5_control`$K4 = zlabels_K4[[1]]
starmap_info$`20180417_BZ5_control`$K5 = zlabels_K5[[1]]
starmap_info$`20180417_BZ5_control`$data_name = "BZ5"

starmap_info$`20180419_BZ9_control`$K3 = zlabels_K3[[2]]
starmap_info$`20180419_BZ9_control`$K4 = zlabels_K4[[2]]
starmap_info$`20180419_BZ9_control`$K5 = zlabels_K5[[2]]
starmap_info$`20180419_BZ9_control`$data_name = "BZ9"

starmap_info$`20180424_BZ14_control`$K3 = zlabels_K3[[3]]
starmap_info$`20180424_BZ14_control`$K4 = zlabels_K4[[3]]
starmap_info$`20180424_BZ14_control`$K5 = zlabels_K5[[3]]
starmap_info$`20180424_BZ14_control`$data_name = "BZ14"

result_df = rbind(starmap_info$`20180417_BZ5_control`, starmap_info$`20180419_BZ9_control`, starmap_info$`20180424_BZ14_control`)
write.csv(result_df, file.path(file_path, "BASS_Results/starmap_K345.csv"))
