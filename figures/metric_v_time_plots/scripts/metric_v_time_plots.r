packages <- c("ggplot2", "dplyr", "patchwork")
for (pkg in packages) {
    suppressPackageStartupMessages(
        suppressWarnings(
            library(
                pkg,
                character.only = TRUE,
                quietly = TRUE,
                warn.conflicts = FALSE
            )
        )
    )
}

umap_file_path <- file.path(
    "../../../data/umap/combined_umap_transformed.parquet"
)
mAP_file_path <- file.path(
    "../../../4.mAP_analysis/data/mAP/mAP_scores_CP_scDINO.parquet"
)
cell_count_file_path <- file.path(
    "../../../2.cell_tracks_data/data/combined_stats.parquet"
)
pca_file_path <- file.path(
    "../../../data/PCA/PCA_2D_combined_features.parquet"
)

# final figure path
figures_path <- file.path("../figures")
if (!dir.exists(figures_path)) {
    dir.create(figures_path, recursive = TRUE)
}
final_figure_path <- file.path(
    figures_path,
    "metric_v_time_plot.png"
)

umap_df <- arrow::read_parquet(umap_file_path)
mAP_df <- arrow::read_parquet(mAP_file_path)
cell_count_df <- arrow::read_parquet(cell_count_file_path)
pca_df <- arrow::read_parquet(pca_file_path)


color_pallete_for_dose <- c(
    "0.0" = "#85FF33",
    "0.61" = "#75FF1A",
    "1.22" = "#62FF00",
    "2.44" = "#4DC507",
    "4.88" = "#398E0B",
    "9.77" = "#265A0C",
    "19.53" = "#132B08",
    "39.06" = "#620B8E",
    "78.13" = "#410C5A",
    "156.25" = "#21082B"
)

umap_df$Metadata_dose <- as.character(umap_df$Metadata_dose)
umap_df$Metadata_dose <- factor(
    umap_df$Metadata_dose,
    levels = c(
        '0.0',
        '0.61',
        '1.22',
        '2.44',
        '4.88',
        '9.77',
        '19.53',
        '39.06',
        '78.13',
        '156.25'
    )
)

mAP_df$Metadata_dose <- as.character(mAP_df$Metadata_dose)
mAP_df$Metadata_dose <- factor(
    mAP_df$Metadata_dose,
    levels = c(
        '0.0',
        '0.61',
        '1.22',
        '2.44',
        '4.88',
        '9.77',
        '19.53',
        '39.06',
        '78.13',
        '156.25'
    )
)
cell_count_df$Metadata_dose <- as.character(cell_count_df$Metadata_dose)
cell_count_df$Metadata_dose <- factor(
    cell_count_df$Metadata_dose,
    levels = c(
        '0',
        '0.61',
        '1.22',
        '2.44',
        '4.88',
        '9.77',
        '19.53',
        '39.06',
        '78.13',
        '156.25'
    )
)
pca_df$Metadata_dose <- as.character(pca_df$Metadata_dose)
pca_df$Metadata_dose <- factor(
    pca_df$Metadata_dose,
    levels = c(
        '0.0',
        '0.61',
        '1.22',
        '2.44',
        '4.88',
        '9.77',
        '19.53',
        '39.06',
        '78.13',
        '156.25'
    )
)
umap_df$Metadata_Time <- as.numeric(umap_df$Metadata_Time) * 30
mAP_df$Metadata_Time <- as.numeric(mAP_df$Metadata_Time) * 30
cell_count_df$Metadata_Time <- as.numeric(cell_count_df$Metadata_Time) * 30
pca_df$Metadata_Time <- as.numeric(pca_df$Metadata_Time) * 30


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

# replace shuffle values with Shuffled and Not Shuffled
mAP_df$Shuffle <- gsub(
    "True",
    "Shuffled",
    mAP_df$Shuffle
)
mAP_df$Shuffle <- gsub(
    "False",
    "Not Shuffled",
    mAP_df$Shuffle
)

mAP_plot <- (
    ggplot(data = mAP_df, aes(x = Metadata_Time, y = mean_average_precision))
    + geom_line(aes(color = Metadata_dose), size = 2)
    + facet_wrap(Shuffle~.)
    + scale_color_manual(values = color_pallete_for_dose)
    + labs(
        x = "Time (minutes)",
        y = "Mean Average Precision (mAP)",
        color = "Dose (nM)",
    )

    # change the legend title
    + guides(
        color = guide_legend(
            override.aes = list(size = 5),
            title.position = "top",
            title.hjust = 0.5,
            title.theme = element_text(size = font_size - 4 ),
            label.theme = element_text(size = font_size - 4),
            nrow = 2,
        ))
    + plot_themes

)
mAP_plot

umap_df$Metadata_dose_w_unit <- paste0(
    umap_df$Metadata_dose,
    " nM"
)
umap_df$Metadata_dose_w_unit <- as.character(umap_df$Metadata_dose_w_unit)
umap_df$Metadata_dose_w_unit <- factor(
    umap_df$Metadata_dose_w_unit,
    levels = c(
        '0.0 nM',
        '0.61 nM',
        '1.22 nM',
        '2.44 nM',
        '4.88 nM',
        '9.77 nM',
        '19.53 nM',
        '39.06 nM',
        '78.13 nM',
        '156.25 nM'
    )
)

temporal_palette <- c(
    "#008CF5", "#0079E7", "#0066D9", "#0053CB", "#0040BD", "#002D9F", "#001A91", "#000781", "#000570", "#000460", "#000350", "#000240", "#000130"
)

umap_df$Metadata_Time <- as.numeric(umap_df$Metadata_Time)
umap_plot_facet <- (
    ggplot(data = umap_df, aes(x = UMAP0, y = UMAP1))
    + geom_point(aes(color = Metadata_Time), size = 0.2, alpha = 0.2)
    + scale_color_gradientn(
        colors = temporal_palette,
        breaks = c(0, 180, 360), # breaks at 0, 90, and 360 minutes
        labels = c("0 min", "180 min", "360 min")
    )
    + labs(
        x = "UMAP 0",
        y = "UMAP 1",
        color = "Time (minutes)",
    )
    + facet_wrap(Metadata_dose_w_unit~., nrow = 2)
    + guides(
        color = guide_colorbar(
            title.position = "top",
            title.hjust = 0.5,
            title.theme = element_text(size = 16),
            # make the legend longer
            barwidth = 20,
        ))
    + plot_themes
    )
umap_plot_facet

# set temporal colour palette of 13 hues of blue
temporal_palette <- c(
    "#008CF5", "#0079E7", "#0066D9", "#0053CB", "#0040BD", "#002D9F", "#001A91", "#000781", "#000570", "#000460", "#000350", "#000240", "#000130"
)
# calculate the centroid of each UMAP cluster dose and time wise
umap_df_centroids <- umap_df %>% group_by(Metadata_dose, Metadata_Time) %>% summarise(
    UMAP0_centroid = mean(UMAP0),
    UMAP1_centroid = mean(UMAP1)
)
umap_df_centroids$Metadata_Time <- as.numeric(gsub(" min", "", umap_df_centroids$Metadata_Time))
umap_df_centroids$Metadata_dose_w_unit <- paste0(
    umap_df_centroids$Metadata_dose,
    " nM"
)
umap_df_centroids$Metadata_dose_w_unit <- as.character(umap_df_centroids$Metadata_dose_w_unit)
umap_df_centroids$Metadata_dose_w_unit <- factor(
    umap_df_centroids$Metadata_dose_w_unit,
    levels = c(
        '0.0 nM',
        '0.61 nM',
        '1.22 nM',
        '2.44 nM',
        '4.88 nM',
        '9.77 nM',
        '19.53 nM',
        '39.06 nM',
        '78.13 nM',
        '156.25 nM'
    )
)


width <- 15
height <- 15
options(repr.plot.width = width, repr.plot.height = height)
# plot the centroids per dose over time
umap_centroid_plot <- (
    ggplot(data = umap_df_centroids, aes(x = UMAP0_centroid, y = UMAP1_centroid, color = Metadata_Time))
    + geom_point(size = 5)
    + theme_bw()
    + labs( x = "UMAP0", y = "UMAP1")
    # add custom colors
    + scale_color_gradientn(
        colors = temporal_palette,

        breaks = c(0, 180, 360), # breaks at 0, 90, and 360 minutes
        labels = c("0 min", "180 min", "360 min"),
        name = "Time (minutes)",
        guide = guide_colorbar(
            title.position = "top",
            title.hjust = 0.5,
            title.theme = element_text(size = 24),
            # make the legend longer
            barwidth = 20
        )
    )
    + theme(
        strip.text.x = element_text(size = 20),
        strip.text.y = element_text(size = 20),
        axis.text.x = element_text(size = 20, angle = 45, hjust = 1),
        axis.text.y = element_text(size = 20),
        axis.title.x = element_text(size = 20),
        axis.title.y = element_text(size = 20),
        axis.ticks.x = element_line(size = 1),
        axis.ticks.y = element_line(size = 1),
        legend.text = element_text(size = 24),
        legend.position = "bottom",
        legend.title = element_text(size = 24, hjust = 0.5),
        plot.title = element_text(size = 24, hjust = 0.5)
        )
    + facet_wrap(~Metadata_dose_w_unit,nrow = 2)

)
umap_centroid_plot


pca_df$Metadata_Time <- as.numeric(gsub(" min", "", pca_df$Metadata_Time))
pca_df$Metadata_dose_w_unit <- paste0(
    pca_df$Metadata_dose,
    " nM"
)
pca_df$Metadata_dose_w_unit <- as.character(pca_df$Metadata_dose_w_unit)
pca_df$Metadata_dose_w_unit <- factor(
    pca_df$Metadata_dose_w_unit,
    levels = c(
        '0.0 nM',
        '0.61 nM',
        '1.22 nM',
        '2.44 nM',
        '4.88 nM',
        '9.77 nM',
        '19.53 nM',
        '39.06 nM',
        '78.13 nM',
        '156.25 nM'
    )
)

head(pca_df)

# set temporal colour palette of 13 hues of blue

# calculate the centroid of each UMAP cluster dose and time wise
pca_df_centroids <- pca_df %>% group_by(Metadata_dose, Metadata_Time) %>% summarise(
    PCA0_centroid = mean(PCA0),
    PCA1_centroid = mean(PCA1)
)
pca_df_centroids$Metadata_Time <- as.numeric(gsub(" min", "", pca_df_centroids$Metadata_Time))
pca_df_centroids$Metadata_dose_w_unit <- paste0(
    pca_df_centroids$Metadata_dose,
    " nM"
)
pca_df_centroids$Metadata_dose_w_unit <- as.character(pca_df_centroids$Metadata_dose_w_unit)
pca_df_centroids$Metadata_dose_w_unit <- factor(
    pca_df_centroids$Metadata_dose_w_unit,
    levels = c(
        '0.0 nM',
        '0.61 nM',
        '1.22 nM',
        '2.44 nM',
        '4.88 nM',
        '9.77 nM',
        '19.53 nM',
        '39.06 nM',
        '78.13 nM',
        '156.25 nM'
    )
)

pca_plot_facet <- (
    ggplot(data = pca_df_centroids, aes(x = PCA0_centroid, y = PCA1_centroid))
    + geom_point(aes(color = Metadata_Time), size = 5)
    + scale_color_gradientn(
        colors = temporal_palette,
        breaks = c(0, 180, 360), # breaks at 0, 90, and 360 minutes
        labels = c("0 min", "180 min", "360 min")
    )
    + labs(
        x = "PCA 0",
        y = "PCA 1",
        color = "Time (minutes)",
    )
    # change the x scale
    + facet_wrap(Metadata_dose_w_unit~., nrow = 2)
    + guides(
        color = guide_colorbar(
            title.position = "top",
            title.hjust = 0.5,
            title.theme = element_text(size = 24),
            # make the legend longer
            barwidth = 20,
        ))
    + plot_themes

    )
pca_plot_facet

# get the well from the well_fov column, get the first part of the string
# before the underscore and number
cell_count_df$well <- sub("_.*", "", cell_count_df$well_fov)
# map the well to the dose from the mAP_df
well_dose_df <- umap_df %>%
    select(well = Metadata_Well, dose = Metadata_dose) %>%
    distinct()
# map the well to the dose in the cell_count_df
cell_count_df <- cell_count_df %>%
    left_join(well_dose_df, by = "well")
# get the metadata well columnd from the well_fov column
cell_count_df$Metadata_Well <- sub("_.*", "", cell_count_df$well_fov)
# convert dose to factor


# get the counts of cells per timepoint per well
cell_count_df <- cell_count_df %>%
    group_by(Metadata_Time, Metadata_Well, Metadata_dose) %>%
    summarise(
        cell_count = sum(total_CP_cells),
        .groups = "drop"
    )

cell_count_norm_df <- cell_count_df %>%
    group_by(Metadata_Well, Metadata_dose) %>%
    mutate(
        baseline_count = first(cell_count[Metadata_Time == min(Metadata_Time)]),
        cell_count_norm = cell_count / baseline_count
    ) %>%
    select(-baseline_count)

cell_count_v_time_plot_colored_by_dose <- (
    ggplot(data = cell_count_df, aes(x = Metadata_Time, y = cell_count))
    + geom_line(aes(group = Metadata_Well,color = Metadata_dose), size = 2)
    + scale_color_manual(values = color_pallete_for_dose)
    + labs(
        x = "Time (minutes)",
        y = "Cell Count",
        color = "Dose (nM)",
    )
    + plot_themes

)
cell_count_v_time_plot_colored_by_dose

normalized_cell_count_v_time_plot_colored_by_dose <- (
    ggplot(data = cell_count_norm_df, aes(x = Metadata_Time, y = cell_count_norm))
    + geom_line(aes(group = Metadata_Well,color = Metadata_dose), size = 2)
    + scale_color_manual(values = color_pallete_for_dose)
    + labs(
        x = "Time (minutes)",
        y = "Normalized Cell Count",
        color = "Dose (nM)",
    )
    + plot_themes

)
normalized_cell_count_v_time_plot_colored_by_dose

width <- 17
height <- 15
options(repr.plot.width=width, repr.plot.height=height)
layout <- c(
    area(t=1, b=1, l=1, r=2), # A
    area(t=1, b=1, l=3, r=4), # B
    area(t=2, b=2, l=1, r=2), # C
    area(t=2, b=2, l=3, r=4) # D
)
metric_v_time_final_plot <- (
    umap_plot_facet
    + umap_centroid_plot
    + cell_count_v_time_plot_colored_by_dose
    + mAP_plot

    + plot_layout(
        design = layout,
        widths = c(0.6, 1)
        )
    # make bottom plot not align
    + plot_annotation(tag_levels = 'A') & theme(plot.tag = element_text(size = 28))
)
ggsave(
    filename = final_figure_path,
    plot = metric_v_time_final_plot,
    width = width,
    height = height,
    dpi = 600
)
metric_v_time_final_plot
