library(ggplot2)

args <- commandArgs(trailingOnly = TRUE)
num_rep <- if (length(args) >= 1) args[[1]] else "1000"

df <- read.csv(sprintf("stratum_cre_B%s.csv", num_rep))

p_levels <- c("1", "0.1", "0.01", "0.001")

cre <- df[df$design == "CRE", ]
non_cre <- df[df$design != "CRE", ]

cre_rep <- cre
cre_rep$design <- "ReP"
cre_rem <- cre
cre_rem$design <- "ReM"

plot_data <- rbind(cre_rep, cre_rem, non_cre)
plot_data$p_label <- ifelse(
  abs(plot_data$accept_prob - 1) < 1e-12,
  "1",
  ifelse(
    abs(plot_data$accept_prob - 0.1) < 1e-12,
    "0.1",
    ifelse(
      abs(plot_data$accept_prob - 0.01) < 1e-12,
      "0.01",
      ifelse(abs(plot_data$accept_prob - 0.001) < 1e-12, "0.001", NA)
    )
  )
)
plot_data$p_label <- factor(plot_data$p_label, levels = p_levels)
plot_data$stratum <- factor(plot_data$stratum)
plot_data$design <- factor(plot_data$design, levels = c("ReM", "ReP"))

write.csv(
  plot_data[, c("stratum", "design", "p_label", "type_I_error", "mcse", "mean_attempts")],
  sprintf("stratum_cre_lines_B%s.csv", num_rep),
  row.names = FALSE
)

p <- ggplot(
  plot_data,
  aes(x = p_label, y = type_I_error, color = stratum, group = stratum)
) +
  geom_hline(yintercept = 0.05, linetype = "dotted", color = "grey35", linewidth = 0.5) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.2) +
  geom_text(
    aes(label = sprintf("%.3f", type_I_error)),
    vjust = -0.75,
    size = 3.0,
    show.legend = FALSE
  ) +
  facet_grid(cols = vars(design)) +
  scale_y_continuous(limits = c(0, 0.25), breaks = seq(0, 0.25, by = 0.05)) +
  labs(
    x = "Rerandomization acceptance probability",
    y = "Type I error",
    color = "Stratum"
  ) +
  theme_grey(base_size = 13) +
  theme(
    legend.position = "top",
    strip.text = element_text(size = 14),
    axis.title = element_text(size = 14),
    panel.spacing = unit(0.14, "in"),
    plot.margin = margin(10, 12, 10, 10)
  )

ggsave(
  sprintf("stratum_cre_lines_B%s.png", num_rep),
  p,
  width = 10.5,
  height = 5.8,
  dpi = 220
)

ggsave(
  sprintf("stratum_cre_lines_B%s.pdf", num_rep),
  p,
  width = 10.5,
  height = 5.8
)
