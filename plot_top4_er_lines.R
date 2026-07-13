library(ggplot2)

load_result <- function(path) {
  env <- new.env(parent = emptyenv())
  load(path, envir = env)
  as.list(env)
}

args <- commandArgs(trailingOnly = TRUE)
num_rep <- if (length(args) >= 1) args[[1]] else "1000"

p_values <- c(0.1, 0.01, 0.001, 0.0001)
p_labels <- c("0.1", "0.01", "0.001", "0.0001")

result_for_p <- function(p) {
  load_result(sprintf("real_data_top4_accelerated_p%g_B%s.Rdata", p, num_rep))
}

results <- setNames(lapply(p_values, result_for_p), p_labels)

er <- function(x) mean(x <= 0.05)

line_data <- do.call(
  rbind,
  lapply(seq_along(p_values), function(i) {
    res <- results[[i]]
    p_lab <- p_labels[[i]]
    data.frame(
      p = p_values[[i]],
      p_label = p_lab,
      analysis = c("FE", "Lin", "Lin"),
      design = c("ReP", "ReP", "ReM"),
      er = c(
        er(res$hacked_p_vec_rep_mn),
        er(res$hacked_p_vec_rep_ss),
        er(res$hacked_p_vec_rem_ss)
      ),
      stringsAsFactors = FALSE
    )
  })
)

baseline <- data.frame(
  analysis = c("FE", "Lin"),
  design = "SRE",
  er = c(
    er(results[[1]]$hacked_p_vec_cre_mn),
    er(results[[1]]$hacked_p_vec_cre_ss)
  ),
  stringsAsFactors = FALSE
)

line_data$p_label <- factor(line_data$p_label, levels = p_labels)
line_data$analysis <- factor(line_data$analysis, levels = c("FE", "Lin"))
baseline$analysis <- factor(baseline$analysis, levels = c("FE", "Lin"))
line_data$label_vjust <- ifelse(line_data$design == "ReP", -0.9, 1.7)

write.csv(
  rbind(
    line_data[, c("analysis", "design", "p_label", "er")],
    transform(baseline, p_label = "baseline")[, c("analysis", "design", "p_label", "er")]
  ),
  sprintf("top4_er_lines_B%s.csv", num_rep),
  row.names = FALSE
)

p <- ggplot(line_data, aes(x = p_label, y = er, color = design, group = design)) +
  geom_hline(yintercept = 0.05, linetype = "dotted", color = "grey35", linewidth = 0.5) +
  geom_hline(
    data = baseline,
    aes(yintercept = er),
    linetype = "dashed",
    color = "grey45",
    linewidth = 0.6
  ) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.2) +
  geom_text(
    aes(label = sprintf("%.3f", er), vjust = label_vjust),
    size = 3.4,
    show.legend = FALSE
  ) +
  geom_text(
    data = baseline,
    aes(x = 1, y = er, label = sprintf("SRE %.3f", er)),
    inherit.aes = FALSE,
    hjust = -0.05,
    vjust = -0.55,
    color = "grey30",
    size = 3.4
  ) +
  facet_grid(rows = vars(analysis)) +
  scale_color_manual(values = c("ReP" = "#2F6FB0", "ReM" = "#B45A2A")) +
  scale_y_continuous(limits = c(0, 0.22), breaks = seq(0, 0.20, by = 0.05)) +
  labs(x = "Rerandomization acceptance probability", y = "ER", color = NULL) +
  theme_grey(base_size = 13) +
  theme(
    legend.position = "top",
    strip.text.y = element_text(size = 14),
    axis.title = element_text(size = 14),
    panel.spacing = unit(0.12, "in"),
    plot.margin = margin(10, 12, 10, 10)
  )

ggsave(
  sprintf("top4_er_lines_B%s.png", num_rep),
  p,
  width = 9.5,
  height = 6.2,
  dpi = 220
)

ggsave(
  sprintf("top4_er_lines_B%s.pdf", num_rep),
  p,
  width = 9.5,
  height = 6.2
)
