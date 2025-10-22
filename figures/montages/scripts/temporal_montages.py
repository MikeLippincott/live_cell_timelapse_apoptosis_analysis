#!/usr/bin/env python
# coding: utf-8

# # This PR plots montages of temporal data for a single cell followed through time.

# ## Imports

# In[2]:


import pathlib
import sys
import tomllib as tomli

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import tifffile
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

sys.path.append("../../../utils")
from scale_bar_util import add_scale_bar

# ## Pathing and preprocessing

# In[4]:


umap_file_path = pathlib.Path(
    "../../../data/umap/combined_umap_transformed.parquet"
).resolve(strict=True)
image_metadata_toml_path = pathlib.Path("../../../utils/pixel_dimension.toml").resolve(
    strict=True
)
large_montage_white_background_output_dir = pathlib.Path(
    "../figures/large_montage_white_background.png"
).resolve()
large_montage_black_background_output_dir = pathlib.Path(
    "../figures/large_montage_black_background.png"
).resolve()
mini_montage_white_background_output_dir = pathlib.Path(
    "../figures/mini_montage_white_background.png"
).resolve()
mini_montage_black_background_output_dir = pathlib.Path(
    "../figures/mini_montage_black_background.png"
).resolve()
mini_montage_black_background_output_dir.parent.mkdir(parents=True, exist_ok=True)

umap_df = pd.read_parquet(umap_file_path)
# make the time column numeric
umap_df["Metadata_Time"] = pd.to_numeric(umap_df["Metadata_Time"])
umap_df["Metadata_Time"] = umap_df["Metadata_Time"].astype(int)
umap_df["Metadata_Time"] = umap_df["Metadata_Time"] * 30
umap_df["Metadata_Well_FOV"] = (
    umap_df["Metadata_Well"].astype(str) + "_" + umap_df["Metadata_FOV"].astype(str)
)

# read the toml
global PIXEL_SIZE_UM
with open(image_metadata_toml_path, "rb") as f:
    PIXEL_SIZE_UM = tomli.load(f)["pixel_size_um"]


# ## Functions

# In[13]:


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

    composite = add_scale_bar(
        image=composite,
        pixel_size_um=PIXEL_SIZE_UM,
        scale_bar_length_um=10,  # um
        scale_bar_height_px=5,  # pixels
        print_text=False,
        padding=10,  # pixels
    )
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


def generate_image_pannel_df(
    df: pd.DataFrame, well_fov: str, cell_id: int
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
        print(f"Error processing well_fov {well_fov}: {e}")
    output_df = pd.DataFrame(output_dict)
    # sort by time
    output_df = output_df.sort_values(by="time")
    output_df.reset_index(drop=True, inplace=True)
    return output_df


# ## Get dfs containing cell tracks

# In[14]:


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


# In[15]:


df = pd.DataFrame(track_ids)
df = df.sort_values(by="well_fovs")
df.head()


# In[16]:


c02_df = generate_image_pannel_df(umap_df, "C-02_0001", 23)  # DMSO
c03_df = generate_image_pannel_df(umap_df, "C-03_0001", 28)  # 0.61
c04_df = generate_image_pannel_df(umap_df, "C-04_0001", 35)  # 1.22
c05_df = generate_image_pannel_df(umap_df, "C-05_0001", 44)  # 2.44
c06_df = generate_image_pannel_df(umap_df, "C-06_0001", 106)  # 4.88
c07_df = generate_image_pannel_df(umap_df, "C-07_0001", 109)  # 9.77
c08_df = generate_image_pannel_df(umap_df, "C-08_0001", 35)  # 19.23
c09_df = generate_image_pannel_df(umap_df, "C-09_0001", 22)  # 39.06
c10_df = generate_image_pannel_df(umap_df, "C-10_0001", 101)  # 78.13
c11_df = generate_image_pannel_df(umap_df, "C-11_0001", 141)  # 156.25


# ## Full montage of all timepoints and all doeses

# In[17]:


for background in ["white", "black"]:
    # create a montage of the images in composite_df
    plt.figure(figsize=(21, 18))
    plt.subplots_adjust(wspace=0.1, hspace=0)
    if background == "white":
        pass
    else:
        # black background
        plt.gcf().set_facecolor("black")
        # text white
        plt.rcParams["text.color"] = "white"
    for index, row in c02_df.iterrows():
        if index == 0:
            plt.subplot(12, len(c02_df) + 1, index + len(c02_df) + 2)
            plt.text(
                0.5, 0.5, f"{row['dose']} nM", fontsize=14, ha="center", va="center"
            )
            plt.axis("off")
            plt.subplot(12, len(c02_df) + 1, index + len(c02_df) + 3)
            plt.imshow(row["composite"])
            plt.title(f"{row['time']} min", fontsize=14)
        else:
            plt.subplot(12, len(c02_df) + 1, index + len(c02_df) + 3)
            plt.imshow(row["composite"])
            plt.title(f"{row['time']} min", fontsize=14)

        plt.axis("off")
    for index, row in c03_df.iterrows():
        if index == 0:
            plt.subplot(12, len(c03_df) + 1, index + 2 * len(c03_df) + 3)
            plt.text(
                0.5, 0.5, f"{row['dose']} nM", fontsize=14, ha="center", va="center"
            )
            plt.axis("off")
            plt.subplot(12, len(c03_df) + 1, index + 2 * len(c03_df) + 4)
            plt.imshow(row["composite"])
        else:
            plt.subplot(12, len(c03_df) + 1, index + 2 * len(c03_df) + 4)
            plt.imshow(row["composite"])
        plt.axis("off")
    for index, row in c04_df.iterrows():
        if index == 0:
            plt.subplot(12, len(c04_df) + 1, index + 3 * len(c04_df) + 4)
            plt.text(
                0.5, 0.5, f"{row['dose']} nM", fontsize=14, ha="center", va="center"
            )
            plt.axis("off")
            plt.subplot(12, len(c04_df) + 1, index + 3 * len(c04_df) + 5)
            plt.imshow(row["composite"])
        else:
            plt.subplot(12, len(c04_df) + 1, index + 3 * len(c04_df) + 5)
            plt.imshow(row["composite"])
        plt.axis("off")
    for index, row in c05_df.iterrows():
        if index == 0:
            plt.subplot(12, len(c05_df) + 1, index + 4 * len(c05_df) + 5)
            plt.text(
                0.5, 0.5, f"{row['dose']} nM", fontsize=14, ha="center", va="center"
            )
            plt.axis("off")
            plt.subplot(12, len(c05_df) + 1, index + 4 * len(c05_df) + 6)
            plt.imshow(row["composite"])
        else:
            plt.subplot(12, len(c05_df) + 1, index + 4 * len(c05_df) + 6)
            plt.imshow(row["composite"])
        plt.axis("off")
    for index, row in c06_df.iterrows():
        if index == 0:
            plt.subplot(12, len(c06_df) + 1, index + 5 * len(c06_df) + 6)
            plt.text(
                0.5, 0.5, f"{row['dose']} nM", fontsize=14, ha="center", va="center"
            )
            plt.axis("off")
            plt.subplot(12, len(c06_df) + 1, index + 5 * len(c06_df) + 7)
            plt.imshow(row["composite"])
        else:
            plt.subplot(12, len(c06_df) + 1, index + 5 * len(c06_df) + 7)
            plt.imshow(row["composite"])
        plt.axis("off")
    for index, row in c07_df.iterrows():
        if index == 0:
            plt.subplot(12, len(c07_df) + 1, index + 6 * len(c07_df) + 7)
            plt.text(
                0.5, 0.5, f"{row['dose']} nM", fontsize=14, ha="center", va="center"
            )
            plt.axis("off")
            plt.subplot(12, len(c07_df) + 1, index + 6 * len(c07_df) + 8)
            plt.imshow(row["composite"])
        else:
            plt.subplot(12, len(c07_df) + 1, index + 6 * len(c07_df) + 8)
            plt.imshow(row["composite"])
        plt.axis("off")
    for index, row in c08_df.iterrows():
        if index == 0:
            plt.subplot(12, len(c08_df) + 1, index + 7 * len(c08_df) + 8)
            plt.text(
                0.5, 0.5, f"{row['dose']} nM", fontsize=14, ha="center", va="center"
            )
            plt.axis("off")
            plt.subplot(12, len(c08_df) + 1, index + 7 * len(c08_df) + 9)
            plt.imshow(row["composite"])
        else:
            plt.subplot(12, len(c08_df) + 1, index + 7 * len(c08_df) + 9)
            plt.imshow(row["composite"])
        plt.axis("off")
    for index, row in c09_df.iterrows():
        if index == 0:
            plt.subplot(12, len(c09_df) + 1, index + 8 * len(c09_df) + 9)
            plt.text(
                0.5, 0.5, f"{row['dose']} nM", fontsize=14, ha="center", va="center"
            )
            plt.axis("off")
            plt.subplot(12, len(c09_df) + 1, index + 8 * len(c09_df) + 10)
            plt.imshow(row["composite"])
        else:
            plt.subplot(12, len(c09_df) + 1, index + 8 * len(c09_df) + 10)
            plt.imshow(row["composite"])
        plt.axis("off")
    for index, row in c10_df.iterrows():
        if index == 0:
            plt.subplot(12, len(c10_df) + 1, index + 9 * len(c10_df) + 10)
            plt.text(
                0.5, 0.5, f"{row['dose']} nM", fontsize=14, ha="center", va="center"
            )
            plt.axis("off")
            plt.subplot(12, len(c10_df) + 1, index + 9 * len(c10_df) + 11)
            plt.imshow(row["composite"])
        else:
            plt.subplot(12, len(c10_df) + 1, index + 9 * len(c10_df) + 11)
            plt.imshow(row["composite"])
        plt.axis("off")
    for index, row in c11_df.iterrows():
        if index == 0:
            plt.subplot(12, len(c11_df) + 1, index + 10 * len(c11_df) + 11)
            plt.text(
                0.5, 0.5, f"{row['dose']} nM", fontsize=14, ha="center", va="center"
            )
            plt.axis("off")
            plt.subplot(12, len(c11_df) + 1, index + 10 * len(c11_df) + 12)
            plt.imshow(row["composite"])
        else:
            plt.subplot(12, len(c11_df) + 1, index + 10 * len(c11_df) + 12)
            plt.imshow(row["composite"])
        plt.axis("off")
    if background == "white":
        plt.savefig(
            large_montage_white_background_output_dir,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.1,
        )
    else:
        plt.savefig(
            large_montage_black_background_output_dir,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.1,
        )
    plt.show()


# ## Mini montage of selected doses

# In[18]:


for background in ["white", "black"]:
    # create a montage of the images in composite_df
    plt.figure(figsize=(21, 9))
    plt.subplots_adjust(wspace=0.1, hspace=0)
    if background == "white":
        # black background
        plt.gcf().set_facecolor("white")
        # text white
        plt.rcParams["text.color"] = "black"
    else:
        # black background
        plt.gcf().set_facecolor("black")
        # text white
        plt.rcParams["text.color"] = "white"
    for index, row in c02_df.iterrows():
        if index == 0:
            plt.subplot(6, len(c02_df) + 1, index + len(c02_df) + 2)
            plt.text(
                0.5, 0.5, f"{row['dose']} nM", fontsize=14, ha="center", va="center"
            )
            plt.axis("off")
            plt.subplot(6, len(c02_df) + 1, index + len(c02_df) + 3)
            plt.imshow(row["composite"])
            plt.title(f"{row['time']} min", fontsize=14)
        else:
            plt.subplot(6, len(c02_df) + 1, index + len(c02_df) + 3)
            plt.imshow(row["composite"])
            plt.title(f"{row['time']} min", fontsize=14)
        plt.axis("off")
    for index, row in c06_df.iterrows():
        if index == 0:
            plt.subplot(6, len(c06_df) + 1, index + 2 * len(c06_df) + 3)
            plt.text(
                0.5, 0.5, f"{row['dose']} nM", fontsize=14, ha="center", va="center"
            )
            plt.axis("off")
            plt.subplot(6, len(c06_df) + 1, index + 2 * len(c06_df) + 4)
            plt.imshow(row["composite"])
        else:
            plt.subplot(6, len(c06_df) + 1, index + 2 * len(c06_df) + 4)
            plt.imshow(row["composite"])
        plt.axis("off")
    for index, row in c08_df.iterrows():
        if index == 0:
            plt.subplot(6, len(c08_df) + 1, index + 3 * len(c08_df) + 4)
            plt.text(
                0.5, 0.5, f"{row['dose']} nM", fontsize=14, ha="center", va="center"
            )
            plt.axis("off")
            plt.subplot(6, len(c08_df) + 1, index + 3 * len(c08_df) + 5)
            plt.imshow(row["composite"])
        else:
            plt.subplot(6, len(c08_df) + 1, index + 3 * len(c08_df) + 5)
            plt.imshow(row["composite"])
        plt.axis("off")
    for index, row in c10_df.iterrows():
        if index == 0:
            plt.subplot(6, len(c10_df) + 1, index + 4 * len(c10_df) + 5)
            plt.text(
                0.5, 0.5, f"{row['dose']} nM", fontsize=14, ha="center", va="center"
            )
            plt.axis("off")
            plt.subplot(6, len(c10_df) + 1, index + 4 * len(c10_df) + 6)
            plt.imshow(row["composite"])
        else:
            plt.subplot(6, len(c10_df) + 1, index + 4 * len(c10_df) + 6)
            plt.imshow(row["composite"])
        plt.axis("off")
    for index, row in c11_df.iterrows():
        if index == 0:
            plt.subplot(6, len(c11_df) + 1, index + 5 * len(c11_df) + 6)
            plt.text(
                0.5, 0.5, f"{row['dose']} nM", fontsize=14, ha="center", va="center"
            )
            plt.axis("off")
            plt.subplot(6, len(c11_df) + 1, index + 5 * len(c11_df) + 7)
            plt.imshow(row["composite"])
        else:
            plt.subplot(6, len(c11_df) + 1, index + 5 * len(c11_df) + 7)
            plt.imshow(row["composite"])
        plt.axis("off")
    if background == "white":
        plt.savefig(
            mini_montage_white_background_output_dir,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.1,
        )
    else:
        plt.savefig(
            mini_montage_black_background_output_dir,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.1,
        )
    plt.show()
