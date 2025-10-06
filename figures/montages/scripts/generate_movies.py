#!/usr/bin/env python
# coding: utf-8

# # This PR plots montages of temporal data for a single cell followed through time.

# ## Imports

# In[1]:


import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import tifffile
from IPython.display import Image as IPyImage
from matplotlib import animation
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from tqdm import tqdm

# ## Functions

# In[2]:


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize the image to the range [0, 255] and convert to uint8.

    Parameters
    ----------
    image : np.ndarray
        Input image to be normalized.

    Returns
    -------
    np.ndarray
        Normalized image in uint8 format.
    """
    image = image.astype(np.float32)
    image -= image.min()
    image /= image.max()
    image *= 255.0
    return image.astype(np.uint8)


# make the composite images
def make_composite_image(
    image1_path: pathlib.Path,  # yellow
    image2_path: pathlib.Path,  # green
    image3_path: pathlib.Path,  # red
    image4_path: pathlib.Path,  # blue
) -> PIL.Image.Image:
    """
    Create a composite image from four input images.

    Parameters
    ----------
    image1_path : pathlib.Path
        Path to the first image (488_1).
    image2_path : pathlib.Path
        Path to the second image (488_2).
    image3_path : pathlib.Path
        Path to the third image (561).
    image4_path : pathlib.Path
        Path to the fourth image (DNA).

    Returns
    -------
    PIL.Image.Image
        Composite image in CYMK format. Where the channels are:
        - C: DNA (blue)
        - M: 488_1 (magenta)
        - Y: 561 (yellow)
        - K: 488_2 (green)
    """
    # Load the images
    image1 = tifffile.imread(image1_path)
    image2 = tifffile.imread(image2_path)
    image3 = tifffile.imread(image3_path)
    image4 = tifffile.imread(image4_path)

    # Normalize the images to the range [0, 255]
    image1 = normalize_image(image1)  # 488_1
    image2 = normalize_image(image2)  # 488_2
    image3 = normalize_image(image3)  # 561
    image4 = normalize_image(image4)  # DNA
    # merge 488_1 and 488_2 into a single green channel by taking the max
    image1 = np.maximum(image1, image2)
    # make a cyan, magenta, yellow composite
    # cyan = green + blue
    # magenta = red + blue
    # yellow = red + green
    # composite = max(cyan, magenta, yellow)
    cyan = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.uint8)
    cyan[..., 1] = image4  # green
    cyan[..., 2] = image4  # blue
    magenta = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.uint8)
    magenta[..., 0] = image1  # red
    magenta[..., 2] = image1  # blue
    yellow = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.uint8)
    yellow[..., 0] = image3  # red
    yellow[..., 1] = image3  # green
    composite = np.maximum(cyan, np.maximum(magenta, yellow))
    # Convert to PIL Image for enhancement
    composite = Image.fromarray(composite)
    enhancer = ImageEnhance.Contrast(composite)
    composite = enhancer.enhance(2)  # Increase contrast
    enhancer = ImageEnhance.Brightness(composite)
    # composite = enhancer.enhance(1.5)  # Increase brightness
    return composite


# scale the images up for better visualization
def scale_image(image: PIL.Image.Image, scale_factor: int = 4) -> PIL.Image.Image:
    """
    Scale the image by a given factor using nearest neighbor interpolation.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image to be scaled.
    scale_factor : int, optional
        Scaling factor, by default 4

    Returns
    -------
    PIL.Image.Image
        Scaled image for better visualization.
    """

    width, height = image.size
    new_size = (int(width * scale_factor), int(height * scale_factor))
    return image.resize(new_size, Image.NEAREST)


def generate_sc_image_panel_df(
    df: pd.DataFrame, well_fov: str, cell_id: int, verbose: bool = False
) -> pd.DataFrame:
    """
    Generate a DataFrame containing composite images for a specific cell over time.


    Parameters
    ----------
    df : pd.DataFrame
        Image-based profile DataFrame.
    well_fov : str
        Well and field of view identifier.
    cell_id : int
        Cell identifier.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: single_cell_id, well_fov, composite, time, dose.
    """
    output_dict = {
        "single_cell_id": [],
        "well_fov": [],
        "composite": [],
        "time": [],
        "dose": [],
    }
    tmp_df = df.loc[
        (df["Metadata_Well_FOV"] == well_fov) & (df["Metadata_track_id"] == cell_id)
    ]
    try:
        for index, row in tmp_df.iterrows():
            well_fov = row["Metadata_Well_FOV"]
            time = row["Metadata_Time"]
            single_cell_id = row["Metadata_track_id"]
            image1_path = pathlib.Path(
                row["Metadata_Image_FileName_CL_488_1_crop"]
            ).resolve(strict=True)
            image2_path = pathlib.Path(
                row["Metadata_Image_FileName_CL_488_2_crop"]
            ).resolve(strict=True)
            image3_path = pathlib.Path(
                row["Metadata_Image_FileName_CL_561_crop"]
            ).resolve(strict=True)
            image4_path = pathlib.Path(row["Metadata_Image_FileName_DNA_crop"]).resolve(
                strict=True
            )
            composite_image = scale_image(
                make_composite_image(
                    image1_path=image1_path,
                    image2_path=image2_path,
                    image3_path=image3_path,
                    image4_path=image4_path,
                )
            )
            output_dict["single_cell_id"].append(single_cell_id)
            output_dict["well_fov"].append(well_fov)
            output_dict["composite"].append(composite_image)
            output_dict["time"].append(time)
            output_dict["dose"].append(row["Metadata_dose"])
    except Exception as e:
        if verbose:
            print(f"Error processing well_fov {well_fov}: {e}")
        pass
    output_df = pd.DataFrame(output_dict)
    # sort by time
    output_df = output_df.sort_values(by="time")
    output_df.reset_index(drop=True, inplace=True)
    return output_df


def generate_whole_FOV_image_panel_df(
    df: pd.DataFrame, well_fov: str, verbose: bool = False
) -> pd.DataFrame:
    """
    Generate a DataFrame containing composite images for a specific cell over time.


    Parameters
    ----------
    df : pd.DataFrame
        Image-based profile DataFrame.
    well_fov : str
        Well and field of view identifier.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: well_fov, composite, time, dose.
    """
    output_dict = {
        "well_fov": [],
        "composite": [],
        "time": [],
        "dose": [],
    }
    tmp_df = df.loc[(df["Metadata_Well_FOV"] == well_fov)].copy()
    tmp_df = tmp_df.sort_values(by="Metadata_Time")
    parent_image_path = pathlib.Path(
        "/home/lippincm/4TB_A/live_cell_timelapse_apoptosis/2.cellprofiler_ic_processing/illum_directory/timelapse/"
    )
    try:
        for index, row in tmp_df.iterrows():
            well_fov_path = pathlib.Path(
                f"{parent_image_path}/20231017ChromaLive_6hr_4ch_MaxIP_{row['Metadata_Well']}_F{row['Metadata_FOV']}"
            ).resolve(strict=True)
            well_fov = row["Metadata_Well_FOV"]
            time = row["Metadata_Time"]
            image1_path = pathlib.Path(
                f"{well_fov_path}/{row['Metadata_Image_FileName_CL_488_1']}"
            ).resolve(strict=True)
            image2_path = pathlib.Path(
                f"{well_fov_path}/{row['Metadata_Image_FileName_CL_488_2']}"
            ).resolve(strict=True)
            image3_path = pathlib.Path(
                f"{well_fov_path}/{row['Metadata_Image_FileName_CL_561']}"
            ).resolve(strict=True)
            image4_path = pathlib.Path(
                f"{well_fov_path}/{row['Metadata_Image_FileName_DNA']}"
            ).resolve(strict=True)
            composite_image = scale_image(
                make_composite_image(
                    image1_path=image1_path,
                    image2_path=image2_path,
                    image3_path=image3_path,
                    image4_path=image4_path,
                )
            )
            output_dict["well_fov"].append(well_fov)
            output_dict["composite"].append(composite_image)
            output_dict["time"].append(time)
            output_dict["dose"].append(row["Metadata_dose"])
    except Exception as e:
        if verbose:
            print(f"Error processing well_fov {well_fov}: {e}")
        pass
    output_df = pd.DataFrame(output_dict)
    # sort by time
    output_df = output_df.sort_values(by="time")
    output_df.reset_index(drop=True, inplace=True)
    return output_df


def generate_gif(
    df: pd.DataFrame,
    save_path: pathlib.Path,
    single_cell_crop_df: bool,
    interval: int,
) -> pathlib.Path:
    """
    Generate a GIF from a DataFrame of composite images.
    Composite images can be from single cell crops or whole well FOVs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing composite images and metadata.
        The composite images should Image objects within the DataFrame.
        Stored as a column named 'composite'.
        The DataFrame should also contain columns for 'time' and 'dose'.
    save_path : pathlib.Path
        Parent directory Path to save the generated GIF.
    single_cell_crop_df : bool
        Whether the DataFrame contains single cell crop images.
        If True, the GIF filename will include the single cell ID.
        If False, the GIF filename will be for whole well FOVs.
    interval : int
        Time interval between frames in the GIF, in milliseconds.

    Returns
    -------
    pathlib.Path
        Path to the generated GIF.
        This is to allow for easy display in a notebook.
    """

    font = ImageFont.load_default(size=20)
    # make a video montage
    frames = []

    # Sort the times to ensure proper order
    sorted_times = sorted(df["time"].unique())

    for time in sorted_times:
        # Get the original frame
        original_frame = df[df["time"] == time]["composite"].values[0]
        frame_copy = original_frame.copy()

        # Create drawing context
        draw = ImageDraw.Draw(frame_copy)

        # Get metadata for this timepoint
        dose = df[df["time"] == time]["dose"].values[0]

        # Create label
        label = f"Time {time} min, Dose {dose} uM"

        # Define text position
        text_x, text_y = 10, 10

        # Get text bounding box
        bbox = draw.textbbox((text_x, text_y), label, font=font)

        # Create background rectangle with padding
        padding = 5
        background_bbox = (
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding,
        )

        # Draw the black background rectangle first
        draw.rectangle(background_bbox, fill=(0, 0, 0))

        # Draw the white text on top
        draw.text((text_x, text_y), label, font=font, fill=(255, 255, 255))

        frames.append(frame_copy)

    if single_cell_crop_df:
        single_cell_id = df["single_cell_id"].values[0]
        video_file_path = pathlib.Path(
            save_path
            / f"sc_montage_well_{df['well_fov'].values[0]}_cell_{single_cell_id}.gif"
        )

    else:
        video_file_path = pathlib.Path(
            save_path / f"well_montage_well_{df['well_fov'].values[0]}.gif"
        )
    frames[0].save(
        video_file_path,
        save_all=True,
        append_images=frames[1:],
        duration=interval,
        loop=0,
    )
    return video_file_path


# ## Pathing and preprocessing

# In[3]:


umap_file_path = pathlib.Path(
    "../../../data/umap/combined_umap_transformed.parquet"
).resolve(strict=True)
sc_video_path = pathlib.Path("../figures/montages_videos/single_cell/")
well_fov_video_path = pathlib.Path("../figures/montages_videos/well_fov/")
figure_gifs = pathlib.Path("../figures/figure_gifs/")
sc_video_path.mkdir(parents=True, exist_ok=True)
well_fov_video_path.mkdir(parents=True, exist_ok=True)
figure_gifs.mkdir(parents=True, exist_ok=True)


umap_df = pd.read_parquet(umap_file_path)
# make the time column numeric
umap_df["Metadata_Time"] = pd.to_numeric(umap_df["Metadata_Time"])
umap_df["Metadata_Time"] = umap_df["Metadata_Time"].astype(int)
umap_df["Metadata_Time"] = umap_df["Metadata_Time"] * 30
umap_df["Metadata_Well_FOV"] = (
    umap_df["Metadata_Well"].astype(str) + "_" + umap_df["Metadata_FOV"].astype(str)
)
INTERVAL = 400  # milliseconds


# ## Get dfs containing cell tracks

# In[4]:


# get a random Metadata_track_id for a few wells
well_fovs = umap_df["Metadata_Well_FOV"].unique()
# for each well we will find a Metadata_track_id that has
# all time points
track_ids = {"well_fovs": [], "track_id": []}
np.random.seed(1)
for well_fov in well_fovs:
    well_df = umap_df[umap_df["Metadata_Well_FOV"] == well_fov]
    track_id_counts = well_df["Metadata_track_id"].value_counts()
    # find a track id that has all time points (48)
    full_track_ids = track_id_counts[track_id_counts == 13].index.tolist()

    for track_id in full_track_ids:
        track_ids["well_fovs"].append(well_fov)
        track_ids["track_id"].append(track_id)


# In[5]:


df = pd.DataFrame(track_ids)
df = df.sort_values(by="well_fovs")
df["well"] = df["well_fovs"].str.split("_").str[0]
# sample at random 5 cell track ids from each well
seed = 0
df = (
    df.groupby("well")
    .apply(lambda x: x.sample(5, random_state=seed), include_groups=False)
    .reset_index(drop=True)
)
df.head()


# In[6]:


for row in tqdm(
    df.itertuples(), total=len(df), desc="Generating single cell crop montages"
):
    well_fov = row.well_fovs
    track_id = row.track_id
    try:
        filename = generate_gif(
            generate_sc_image_panel_df(umap_df, well_fov=well_fov, cell_id=track_id),
            save_path=sc_video_path,
            single_cell_crop_df=True,
            interval=INTERVAL,
        )
    except Exception as e:
        pass


# ## Whole well FOV montages

# In[7]:


# well_fovs_df
well_fov_only_images = umap_df.drop_duplicates(
    subset=["Metadata_Well_FOV", "Metadata_Time"], keep="first"
)

# drop rows that do no contain time, dose, well, fov, or Metadata_Image_FileName
columns_to_keep = [
    "Metadata_Time",
    "Metadata_dose",
    "Metadata_Well",
    "Metadata_FOV",
    "Metadata_Well_FOV",
]
columns_to_keep += [
    col
    for col in umap_df.columns
    if "Metadata_Image_FileName" in col and "crop" not in col
]
well_fov_only_images = well_fov_only_images[columns_to_keep]
well_fov_only_images.sort_values(
    by=["Metadata_Well_FOV", "Metadata_Time"], inplace=True
)
well_fov_only_images = well_fov_only_images.reset_index(drop=True)


# In[8]:


for well_fov in tqdm(
    well_fov_only_images["Metadata_Well_FOV"].unique(),
    total=len(well_fov_only_images["Metadata_Well_FOV"].unique()),
    desc="Generating whole well FOV montages",
):
    # show the gif
    filename = generate_gif(
        generate_whole_FOV_image_panel_df(well_fov_only_images, well_fov),
        save_path=well_fov_video_path,
        single_cell_crop_df=False,
        interval=INTERVAL,
    )


# ## umap montages

# In[9]:


for dose in umap_df["Metadata_dose"].unique():
    frames = []
    fig, ax = plt.subplots(figsize=(6, 6))

    tmp_df = umap_df[umap_df["Metadata_dose"] == dose].copy()
    tmp_df.sort_values(by=["Metadata_Time"], inplace=True)
    classes = tmp_df["Metadata_Time"].unique()
    # split the data into n different dfs based on the classes
    dfs = [tmp_df[tmp_df["Metadata_Time"] == c] for c in classes]
    for i in range(len(dfs)):
        df = dfs[i]
        # split the data into the Metadata and the Features
        metadata_columns = df.columns[df.columns.str.contains("Metadata")]
        metadata_df = df[metadata_columns]
        features_df = df.drop(metadata_columns, axis=1)
        dfs[i] = features_df
    # plot the list of dfs and animate them
    ax.set_xlim(min(umap_df["UMAP0"]) - 1, max(umap_df["UMAP0"]) + 1)
    ax.set_ylim(min(umap_df["UMAP1"]) - 1, max(umap_df["UMAP1"]) + 1)
    scat = ax.scatter([], [], c="b", s=1)
    text = ax.text(-4.5, -0.25, "", ha="left", va="top")
    # add title
    ax.set_title(f"Staurosporine {dose} nM")
    # axis titles
    ax.set_xlabel("UMAP0")
    ax.set_ylabel("UMAP1")

    def animate(i):
        df = dfs[i]
        i = i * 30
        scat.set_offsets(df.values)
        text.set_text(f"{i} minutes.")
        return (scat,)

    anim = animation.FuncAnimation(
        fig, init_func=None, func=animate, frames=len(dfs), interval=INTERVAL
    )
    anim.save(f"{figure_gifs}/Staurosporine_{dose}nM.gif")

    plt.close(fig)
