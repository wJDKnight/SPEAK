args <- commandArgs(trailingOnly = TRUE)
data_name <- if (length(args) >= 1) args[[1]] else "BZ14"
starmap_root <- Sys.getenv("LLM_ST_STARMAP_ROOT", "examples/starmap")
path_template <- Sys.getenv("LLM_ST_STARMAP_TEMPLATE", "{sample}")
sample_dir <- gsub("\\{sample\\}", data_name, path_template)
file_path <- file.path(starmap_root, sample_dir)

load(file.path(file_path, "starmap_mpfc.RData"))
object_name <- paste0("20180424_", data_name, "_control")
if (!object_name %in% names(starmap_info) || !object_name %in% names(starmap_cnts)) {
  stop("STARmap object not found in starmap_mpfc.RData: ", object_name)
}

write.csv(starmap_info[[object_name]]["c"], file.path(file_path, "celltype.csv"))
write.csv(starmap_info[[object_name]][c("x", "y")], file.path(file_path, "pos.csv"))
write.csv(starmap_info[[object_name]]["z"], file.path(file_path, "domain.csv"))
write.csv(t(starmap_cnts[[object_name]]), file.path(file_path, "data.csv"))
