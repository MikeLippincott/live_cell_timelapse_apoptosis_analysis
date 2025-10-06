suppressPackageStartupMessages(suppressWarnings({
    library("ggplot2")
    library(dplyr)
    library(tidyr)
    library(ComplexHeatmap)
    library(tibble)
    library(RColorBrewer)
    library(scales)
    library(circlize)
    library(patchwork)
}))
source("../../../utils/r_themes.r")


profile_file_path <- file.path("../../../data/CP_scDINO_features/combined_CP_scDINO_norm_fs_aggregated.parquet")
figure_path <- file.path("../figures/")
if (!dir.exists(figure_path)) {
    dir.create(figure_path, recursive = TRUE)
}

cell_count_file_path <- file.path(
    "../../../2.cell_tracks_data/data/combined_stats.parquet"
)
df <- arrow::read_parquet(profile_file_path) %>% arrange(Metadata_Well)
cell_count_df <- arrow::read_parquet(cell_count_file_path)
# transform the data to standard scalar (-1, 1) format
for (i in 1:ncol(df)) {
    # make sure the column is not metadata
    if (grepl("Metadata_", colnames(df)[i])) {
        next
    }
    if (is.numeric(df[[i]])) {
        df[[i]] <- rescale(df[[i]], to = c(-1, 1))
    }
}
# map each of the Time points to the actual timepoint
df$Metadata_Time <- as.numeric(df$Metadata_Time) * 60 / 2
head(df)

# Drop CP features
df <- df %>%
    select(-ends_with("_CP"))
head(df)

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
cell_count_df$Metadata_Time <- as.numeric(cell_count_df$Metadata_Time) * 30


# get the well from the well_fov column, get the first part of the string
# before the underscore and number
cell_count_df$well <- sub("_.*", "", cell_count_df$well_fov)
cell_count_df$Metadata_Well <- sub("_.*", "", cell_count_df$well_fov)

# get the counts of cells per timepoint per well
cell_count_df <- cell_count_df %>%
    group_by(Metadata_Time, Metadata_Well, Metadata_dose) %>%
    summarise(
        cell_count = sum(total_CP_cells),
        .groups = "drop"
    )

# merge the cell count df with the profile df to get the dose information
df <- df %>%
    left_join(
        cell_count_df %>% select(Metadata_Well, Metadata_Time,cell_count) %>% distinct(),
        by = c("Metadata_Well", "Metadata_Time")
    )
df$Metadata_cell_count <- df$cell_count
# drop cell_count column
df <- df %>% select(-cell_count)
# sort by Metadata_Well, Metadata_dose, Metadata_Time
df <- df %>% arrange(Metadata_Well, Metadata_dose, Metadata_Time)


metadata_cols <- grep("Metadata_", colnames(df), value = TRUE)

# complex heatmap does not compare across heatmaps the scale so we must set it manually
# for more information see:
# https://github.com/jokergoo/EnrichedHeatmap/issues/7

# we will set the color scale the same way that ComplexHeatmap does
# The automatically generated colors map from the minus and plus 99^th of
# the absolute values in the matrix.


global_across_dose_99th_min <- df %>%
    select(-metadata_cols) %>%
    summarise(across(everything(), ~ quantile((.), 0.01, na.rm = TRUE))) %>%
    unlist() %>%
    min(na.rm = TRUE)
global_across_dose_99th_max <- df %>%
    select(-metadata_cols) %>%
    summarise(across(everything(), ~ quantile((.), 0.99, na.rm = TRUE))) %>%
    unlist() %>%
    max(na.rm = TRUE)

print(global_across_dose_99th_min)
print(global_across_dose_99th_max)
col_fun = circlize::colorRamp2(c(global_across_dose_99th_min, 0, global_across_dose_99th_max), c("blue","white", "red"))

# get the list of features
features <- colnames(df)
metadata_cols <- grep("Metadata_", colnames(df), value = TRUE)
features <- features[!features %in% metadata_cols]
features <- as.data.frame(features)
# temporary
# drop all rows that do not contain scDINO in Extra3
# features <- features %>% filter(grepl("scDINO", features))
# remove channel from string in features col
features <- features %>%
    mutate(
        features = gsub("channel_", "", features),
        features = gsub("channel", "", features, fixed = TRUE)
        )

# split the features by _ into multiple columns
features <- features %>%
    separate(features, into = c("Compartment", "Measurement", "Metric", "Extra", "Extra1", "Extra2", "Extra3"), sep = "_", extra = "merge", fill = "right")

# align the scDINO features with the CP features
# if scDINO is in Extra2 then move the values of the Measurement to Extra
features <- features %>%
    mutate(
        Extra = ifelse(grepl("scDINO", Extra1), Compartment, Extra)
    )
features <- features %>%
    mutate(
        Compartment = ifelse(grepl("scDINO", Extra1), Extra1, Compartment)
    )
features <- features %>%
    mutate(
        Measurement = ifelse(grepl("scDINO", Extra1), Extra1, Measurement)
    )

# add CL to Extra if Extra1 is 488 or 561
features$Extra[features$Extra == "488-1"] <- paste0("CL_", features$Extra[features$Extra == "488-1"])
features$Extra[features$Extra == "488-2"] <- paste0("CL_", features$Extra[features$Extra == "488-2"])
features$Extra[features$Extra == "561"] <- paste0("CL_", features$Extra[features$Extra == "561"])


# clean up the features columns
# if Extra is NA then replace with None
features$Extra[is.na(features$Extra)] <- "None"
# if extra is a number then replace with None
features$Extra[grepl("^[0-9]+$", features$Extra)] <- "None"
# replace all other NAs with None
features$Extra1[is.na(features$Extra1)] <- "None"
features$Extra2[is.na(features$Extra2)] <- "None"
# change extra to None if X or Y
features$Extra[features$Extra == "X"] <- "None"
features$Extra[features$Extra == "Y"] <- "None"
# drop the Adjacent channel
features$Extra[features$Extra == "Adjacent"] <- "None"
# if extra1 is 488 then add extra2 to Extra1
features$Extra1[features$Extra1 == "488"] <- paste0(features$Extra1[features$Extra1 == "488"], "_", features$Extra2[features$Extra1 == "488"])
# if extra1 id CL then add extra1 to Extra
features$Extra[features$Extra == "CL"] <- paste0(features$Extra[features$Extra == "CL"], "_", features$Extra1[features$Extra == "CL"])


features <- features %>%
    rename(Channel = Extra) %>%
    select(-Extra1, -Extra2)
# rename channel names to replace "_" with " "
features$Channel <- gsub("CL_488_1", "CL 488_1", features$Channel)
features$Channel <- gsub("CL_488_2", "CL 488_2", features$Channel)
features$Channel <- gsub("CL_488-1", "CL 488_1", features$Channel)
features$Channel <- gsub("CL_488-2", "CL 488_2", features$Channel)
features$Channel <- gsub("CL_561", "CL 561", features$Channel)
features$Channel <- gsub("CP", "None", features$Channel)


# time color function
time_col_fun = colorRamp2(
    c(min(unique(df$Metadata_Time)), max(unique(df$Metadata_Time))), c("white", "purple")
    )

cell_counts <- df$Metadata_cell_count
cell_count_col_fun = colorRamp2(
    c(min(cell_counts, na.rm = TRUE), max(cell_counts, na.rm = TRUE)), c("white", "darkgreen")
    )

column_anno <- HeatmapAnnotation(
    # make the annotation on the bottom

    Time = unique(df$Metadata_Time),
    show_legend = TRUE,
    annotation_name_gp = gpar(fontsize = 2),
    annotation_legend_param = list(
        title_position = "topcenter",
        title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 1.0),
        labels_gp = gpar(fontsize = 16,
        title = gpar(fontsize = 16))
        ),

    col = list(
        Time = time_col_fun
    )
)
row_channel = rowAnnotation(
    Channel = features$Channel,
        annotation_legend_param = list(
        title_position = "topcenter",
        title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 0.5),
        labels_gp = gpar(fontsize = 16,
        # make annotation bar text bigger
        legend = gpar(fontsize = 16),
        annotation_name = gpar(fontsize = 16),
        legend_height = unit(20, "cm"),
        legend_width = unit(1, "cm"),
        # make legend taller
        legend_height = unit(10, "cm"),
        legend_width = unit(1, "cm"),
        legend_key = gpar(fontsize = 16)
        )
    ),



    annotation_name_side = "top",
    # make font size bigger
    annotation_name_gp = gpar(fontsize = 16),
    col = list(
    Channel = c(
            "DNA" = "#0000AB",
            "CL 488_1" = "#B000B0",
            "CL 488_2" = "#00D55B",
            "CL 561" = "#FFFF00",
            "None" = "#B09FB0")
    )
)
row_annotations = c(row_channel)


list_of_mats_for_heatmaps <- list()
list_of_heatmaps <- list()
heatmap_list <- NULL
ht_opt(RESET = TRUE)
ht_opt$message = FALSE
df$Metadata_dose <- as.numeric(df$Metadata_dose)
for (dose in unique(df$Metadata_dose)) {
    # check if the last in the number of doses

    # get the first dose
    single_dose_df <- df %>%
        filter(Metadata_dose == dose) %>%
        group_by(Metadata_Time) %>%
        select(-Metadata_Well, -Metadata_dose, -Metadata_control, -Metadata_compound,-Metadata_number_of_singlecells,-Metadata_plate, -Metadata_cell_count) %>%
        summarise(across(everything(), ~ mean(., na.rm = TRUE))) %>%
        ungroup()

    # sort the columns by Metadata_Time
    single_dose_df <- single_dose_df %>%
        select(Metadata_Time, everything()) %>%
        arrange(Metadata_Time)

    mat <- t(as.matrix(single_dose_df))

    colnames(mat) <- single_dose_df$Metadata_Time
    mat <- mat[-1,]

    if (dose == max(unique(df$Metadata_dose))) {

        heatmap_plot <- Heatmap(
            mat,
            col = col_fun,
            show_row_names = FALSE,
            show_column_names = FALSE,
            cluster_columns = FALSE,
            column_names_gp = gpar(fontsize = 16), # Column name label formatting
            row_names_gp = gpar(fontsize = 14),

            show_heatmap_legend = TRUE,
            heatmap_legend_param = list(
                        title = "Feature\nValue",
                        title_position = "topcenter",
                        # direction = "horizontal",
                        title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 1.0),
                        labels_gp = gpar(fontsize = 16),
                        legend_height = unit(4, "cm"),
                        legend_width = unit(3, "cm"),
                        annotation_legend_side = "bottom"
                        ),
            row_dend_width = unit(2, "cm"),
            column_title = paste0(dose," uM"),
            # add the row annotations
            right_annotation = row_annotations,
            top_annotation = column_anno
        )
    } else {
        heatmap_plot <- Heatmap(
            mat,
            col = col_fun,
            show_row_names = FALSE,
            cluster_columns = FALSE,
            show_column_names = FALSE,
            column_names_gp = gpar(fontsize = 16), # Column name label formatting
            row_names_gp = gpar(fontsize = 14),

            show_heatmap_legend = FALSE,
            heatmap_legend_param = list(
                        title = "Feature\nValue",
                        title_position = "topcenter",
                        title_gp = gpar(fontsize = 16, angle = 0, fontface = "bold", hjust = 1.0),
                        labels_gp = gpar(fontsize = 16),
                        legend_height = unit(4, "cm"),
                        legend_width = unit(3, "cm"),
                        annotation_legend_side = "bottom"
                        ),
            row_dend_width = unit(2, "cm"),
            column_title = paste0(dose," uM"),
            top_annotation = column_anno
        )
    }
    # add the heatmap to the list
    heatmap_list <- heatmap_list + heatmap_plot
}

width <- 17
height <- 15
options(repr.plot.width=width, repr.plot.height=height)
# save the figure
png(
    filename = file.path(figure_path, "combined_CP_scDINO_aggregated_heatmap.png"),
    width = width,
    height = height,
    units = "in",
    res = 600
)
draw(
    heatmap_list,
        merge_legends = TRUE,

)
dev.off()
draw(
    heatmap_list,
        merge_legends = TRUE,
)
