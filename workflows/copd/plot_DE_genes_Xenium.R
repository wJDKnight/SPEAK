library(tidyverse)
library(ggrepel)

copd_root <- Sys.getenv("LLM_ST_COPD_ROOT", "examples/copd/230267_Slide2")
figure_root <- file.path(Sys.getenv("LLM_ST_RUN_ROOT", "."), "figures")
dir.create(figure_root, showWarnings = FALSE, recursive = TRUE)

# Full Slide2 SMC differential-expression statistics exported by
# export_slide2_figure_source.R.  No mixed-slide Seurat object is required.
de_markers <- read_csv(
  file.path(copd_root, "230267_Slide2_SMC_de_full.csv"),
  show_col_types = FALSE
) %>%
  mutate(gene_label = if_else(significant, gene, ""))

p_volcano <- ggplot(
  de_markers,
  aes(x = avg_log2FC, y = -log10(p_val), color = significant)
) +
  geom_point(alpha = 0.8) +
  scale_color_manual(values = c(`TRUE` = "red", `FALSE` = "grey")) +
  geom_text_repel(
    aes(label = gene_label),
    max.overlaps = 20,
    box.padding = 0.5,
    point.padding = 0.3,
    segment.color = "grey50",
    size = 3
  ) +
  labs(
    title = "SMC: Airway Inflamed vs Large Vessel Inflamed",
    x = "Log2 fold change",
    y = "-Log10(p-value)",
    color = "Significant"
  ) +
  geom_vline(xintercept = c(-2, 2), linetype = "dashed", color = "blue", alpha = 0.5) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "blue", alpha = 0.5) +
  theme_minimal() +
  theme(legend.position = "top", plot.title = element_text(size = 8, hjust = 0.5))

ggsave(
  file.path(figure_root, "230267_Slide2_SMC_volcano.pdf"),
  p_volcano,
  width = 5,
  height = 5
)

# Machine-readable marker expression and coordinates for the selected Slide2
# cells underlying the spatial marker panel.
marker_source <- read_csv(
  file.path(copd_root, "Slide2_marker_expression_source.csv"),
  show_col_types = FALSE
)
marker_long <- marker_source %>%
  pivot_longer(
    cols = any_of(c("FOXJ1", "TP63", "MUC5AC", "VWF", "ACTA2")),
    names_to = "gene",
    values_to = "expression"
  )

p_marker <- ggplot(marker_long, aes(x = x, y = y, color = expression)) +
  geom_point(size = 0.25) +
  facet_wrap(~gene, nrow = 1) +
  scale_color_viridis_c() +
  coord_fixed() +
  theme_void() +
  theme(legend.position = "bottom")

ggsave(
  file.path(figure_root, "markergenes_Rplot.pdf"),
  p_marker,
  width = 10,
  height = 3
)
