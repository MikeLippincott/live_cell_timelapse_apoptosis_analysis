suppressPackageStartupMessages(suppressWarnings(library(ggplot2)))
suppressPackageStartupMessages(suppressWarnings(library(dplyr)))
suppressPackageStartupMessages(suppressWarnings(library(argparse)))
source("../../utils/r_themes.r")

data_mode <- "terminal"

# set paths
umap_file_path <- file.path("../../data/umap/",paste0(data_mode,"_umap_transformed.parquet"))
umap_file_path <- normalizePath(umap_file_path)
figures_path <- file.path(paste0("../figures/",data_mode,"/"))
if (!dir.exists(figures_path)) {
  dir.create(figures_path)
}

umap_df <- arrow::read_parquet(umap_file_path)


# make the dose a factor with levels
umap_df$Metadata_dose <- factor(umap_df$Metadata_dose, levels = c(
    "0",
    "0.61",
    "1.22",
    "2.44",
    "4.88",
    "9.77",
    "19.53",
    "39.06",
    "78.13",
    "156.25"
    )
    )



# make a ggplot of the umap
width <- 10
height <- 10
options(repr.plot.width = width, repr.plot.height = height)
umap_plot <- (
    ggplot(data = umap_df, aes(x = UMAP0, y = UMAP1, color = Metadata_dose))
    + geom_point(size = 0.9, alpha = 0.8)
    + theme_bw()

    + labs( x = "UMAP0", y = "UMAP1")
    + theme(
        strip.text.x = element_text(size = 18),
        strip.text.y = element_text(size = 18),
        axis.text.x = element_text(size = 18),
        axis.text.y = element_text(size = 18),
        axis.title.x = element_text(size = 24),
        axis.title.y = element_text(size = 24),
        legend.title = element_text(size = 20),
        legend.text = element_text(size = 20),
        legend.position = "bottom",
        legend.box = "horizontal",


        )
    + scale_color_manual(values = color_palette_dose)
    + guides(
        color = guide_legend(
            override.aes = list(size = 5),
            title = "Dose (nM)",
            title.position = "top",
            title.hjust = 0.5,
            # make them horizontal
            nrow = 2,

        )
    )


)
umap_plot
# save
ggsave(paste0("../figures/",data_mode,"/umap_plot_dose.png"), plot = umap_plot, width = width, height = height, dpi = 600)
