library(ggplot2)

load_result <- function(path) {
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  as.list(env)
}

make_block <- function(analysis, design, pvals) {
  data.frame(
    p = pvals,
    analysis = analysis,
    design = design,
    stringsAsFactors = FALSE
  )
}

result_p1 <- load_result("real_data_top4_accelerated_p0.1_B1000.Rdata")
result_p01 <- load_result("real_data_top4_accelerated_p0.01_B1000.Rdata")
result_p001 <- load_result("real_data_top4_accelerated_p0.001_B1000.Rdata")

design_levels <- c(
  "SRE",
  "ReP (p=0.1)",
  "ReP (p=0.01)",
  "ReP (p=0.001)",
  "ReM (p=0.1)",
  "ReM (p=0.01)",
  "ReM (p=0.001)"
)
analysis_levels <- c("FE", "Lin")

plot_data <- rbind(
  make_block("FE", "SRE", result_p1$hacked_p_vec_cre_mn),
  make_block("FE", "ReP (p=0.1)", result_p1$hacked_p_vec_rep_mn),
  make_block("FE", "ReP (p=0.01)", result_p01$hacked_p_vec_rep_mn),
  make_block("FE", "ReP (p=0.001)", result_p001$hacked_p_vec_rep_mn),
  make_block("Lin", "SRE", result_p1$hacked_p_vec_cre_ss),
  make_block("Lin", "ReP (p=0.1)", result_p1$hacked_p_vec_rep_ss),
  make_block("Lin", "ReP (p=0.01)", result_p01$hacked_p_vec_rep_ss),
  make_block("Lin", "ReP (p=0.001)", result_p001$hacked_p_vec_rep_ss),
  make_block("Lin", "ReM (p=0.1)", result_p1$hacked_p_vec_rem_ss),
  make_block("Lin", "ReM (p=0.01)", result_p01$hacked_p_vec_rem_ss),
  make_block("Lin", "ReM (p=0.001)", result_p001$hacked_p_vec_rem_ss)
)

plot_data$design <- factor(plot_data$design, levels = design_levels)
plot_data$analysis <- factor(plot_data$analysis, levels = analysis_levels)

er_data <- aggregate(
  p ~ analysis + design,
  data = plot_data,
  FUN = function(x) mean(x <= 0.05)
)
names(er_data)[names(er_data) == "p"] <- "er"
er_data$label <- sprintf("ER = %.3f", er_data$er)
er_data$x <- 0.66
er_data$y <- 4.65

summary_table <- er_data[, c("analysis", "design", "er")]
summary_table <- summary_table[order(summary_table$analysis, summary_table$design), ]
write.csv(summary_table, "top4_simulation_er_summary_B1000.csv", row.names = FALSE)

p <- ggplot(plot_data, aes(x = p)) +
  geom_histogram(
    aes(y = after_stat(density)),
    bins = 120,
    boundary = 0,
    fill = "grey65",
    color = "grey65"
  ) +
  geom_hline(yintercept = 1, linetype = "dashed", linewidth = 0.4) +
  geom_text(
    data = er_data,
    aes(x = x, y = y, label = label),
    inherit.aes = FALSE,
    color = "red",
    size = 6
  ) +
  facet_grid(
    rows = vars(analysis),
    cols = vars(design),
    drop = FALSE
  ) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 4.85), expand = FALSE) +
  scale_x_continuous(breaks = c(0, 0.5, 1)) +
  labs(x = NULL, y = "Density") +
  theme_grey(base_size = 13) +
  theme(
    strip.text = element_text(size = 14),
    axis.title.y = element_text(size = 18),
    axis.text = element_text(size = 9),
    panel.spacing = unit(0.12, "in"),
    plot.margin = margin(10, 10, 10, 10)
  )

ggsave(
  "top4_simulation_pvalue_density_B1000.png",
  p,
  width = 19,
  height = 7.2,
  dpi = 220
)

ggsave(
  "top4_simulation_pvalue_density_B1000.pdf",
  p,
  width = 19,
  height = 7.2
)
