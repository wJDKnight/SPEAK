library(ggplot2)
library(ggalluvial)
library(dplyr)

copd_root <- Sys.getenv("LLM_ST_COPD_ROOT", "examples/copd/230267_Slide2")
figure_root <- file.path(Sys.getenv("LLM_ST_RUN_ROOT", "."), "figures")

# Create figures directory
dir.create(figure_root, showWarnings = FALSE, recursive = TRUE)

# Read data
df <- read.csv(file.path(copd_root, "230267_Slide2_finetune_gpt4o_refined_k20.csv"))

# Prepare data for plotting
# Count frequencies of pairs
plot_data <- df %>%
  group_by(zeroshot_gpt4o_mini_refined, finetune_gpt4o_mini_refined) %>%
  summarise(Freq = n(), .groups = "drop") %>%
  rename(axis1 = zeroshot_gpt4o_mini_refined, axis2 = finetune_gpt4o_mini_refined)

# Define colors
colors <- c(
  'Airway Inflamed' = '#e377c2',
  'Alveolar' = '#ff7f0e',
  'Fibrotic' = '#8c564b',
  'Immune' = '#d62728',
  'Inflamed' = '#7f7f7f',
  'Large Vessel Healthy' = '#9467bd',
  'Large Vessel Inflamed' = '#2ca02c',
  'Repair' = '#1f77b4'
)

# Create Plot
p <- ggplot(plot_data,
       aes(y = Freq, axis1 = axis2, axis2 = axis1)) +
  geom_alluvium(aes(fill = axis1), width = 1/12) +
  geom_stratum(aes(fill = after_stat(stratum)), width = 1/12, size = 0) +
  # geom_label(stat = "stratum", aes(label = after_stat(stratum)), size = 3) +
  scale_x_discrete(limits = c("axis1", "axis2"), expand = c(.05, .05), labels = c("Zeroshot", "Finetune")) +
  scale_fill_manual(values = colors) +
  coord_flip() +
  theme_void() + 
  theme(legend.position = "bottom") +
  ggtitle("Refinement Flow: Zeroshot vs Finetune")

# Save plot
output_path <- file.path(figure_root, "sankey_plot.pdf")
ggsave(output_path, plot = p, width = 3.5, height = 3, dpi = 300)

message("Plot saved to ", output_path)
