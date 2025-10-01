suppressPackageStartupMessages(
        suppressWarnings(
            library(viridis)
        )
)
# color palette for plotting dose response curves

color_palette_dose <- c(
    "0" = "#132B08", #132B08
    "0.61" = "#265A0C", #265A0C
    "1.22" = "#398E0B", #398E0B
    "2.44" = "#4DC507", #4DC507
    "4.88" = "#62FF00", #62FF00
    "9.77" = "#75FF1A", #75FF1A
    "19.53" = "#85FF33", #85FF33
    "39.06" = "#620B8E", #620B8E
    "78.13" = "#410C5A", #410C5A
    "156.25" = "#21082B" #21082B
)

turbo_colors <- viridis::turbo(10)
color_palette_dose_turbo <- c(
    "0" = turbo_colors[1],
    "0.61" = turbo_colors[2],
    "1.22" = turbo_colors[3],
    "2.44" = turbo_colors[4],
    "4.88" = turbo_colors[5],
    "9.77" = turbo_colors[6],
    "19.53" = turbo_colors[7],
    "39.06" = turbo_colors[8],
    "78.13" = turbo_colors[9],
    "156.25" = turbo_colors[10]
)
temporal_palette <- c(
    "#008CF5", "#0079E7", "#0066D9", "#0053CB", "#0040BD", "#002D9F", "#001A91", "#000781", "#000570", "#000460", "#000350", "#000240", "#000130"
)

dose_guides_color <- guides(
        color = guide_legend(
            override.aes = list(size = 7, alpha = 1),
            title.position = "top",
            title.hjust = 0.5,
            )
    )
shuffle_guides_shape <- guides(
        shape = guide_legend(
            override.aes = list(size = 7, alpha = 1),
            title.position = "top",
            title.hjust = 0.5,
            nrow = 2,
        )
    )
font_size <- 24
plot_themes <- (
    theme_bw()
    + theme(
        legend.position = "bottom",
        legend.text = element_text(size = font_size),
        legend.title = element_text(size = font_size),
        axis.title.x = element_text(size = font_size),
        axis.text.x = element_text(size = font_size),
        axis.title.y = element_text(size = font_size),
        axis.text.y = element_text(size = font_size),
        strip.text = element_text(size = font_size -2),
    )
)
