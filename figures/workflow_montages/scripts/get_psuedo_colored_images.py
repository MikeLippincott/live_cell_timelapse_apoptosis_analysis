#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import pathlib

import numpy as np
import pandas as pd
import PIL
import tifffile
from PIL import Image, ImageDraw, ImageEnhance, ImageFont  # import ImageEnhance

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False
if in_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# In[ ]:


# set the arg parser
parser = argparse.ArgumentParser(description="UMAP on a matrix")

parser.add_argument(
    "--composite_or_single_channel",
    type=str,
    default="False",
    help="Whether to use composite images (True) or single channel images (False)",
    choices=["True", "False"],
)

# get the args
args = parser.parse_args()

# set data mode to either "CP" or "scDINO" or "combined" or "terminal"
composite_or_single_channel = args.composite_or_single_channel
print(f"composite_or_single_channel: {composite_or_single_channel}")
if composite_or_single_channel == "True":
    composite = True
    pseudo_color = False
else:
    composite = False
    pseudo_color = True
print(f"composite: {composite}, pseudo_color: {pseudo_color}")


# ## Functions

# In[ ]:


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
    try:
        image /= image.max()
    except ZeroDivisionError:
        raise ValueError("Maximum value of the image is zero, cannot normalize.")
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
        - M: 488_1 (magenta) + 488_2 (magenta)
        - Y: 561 (yellow)
        - K: Not used"""
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
    return composite


def make_pseudo_color_image(
    image1_path: pathlib.Path,  # yellow
    image2_path: pathlib.Path,  # green
    image3_path: pathlib.Path,  # red
    image4_path: pathlib.Path,  # blue
) -> PIL.Image.Image:
    """
    Create multiple pseudo-colored images from four input images.

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
        pseudo color image. Where the channels are:
        - DNA (cyan)
        - 488_1 (magenta)
        - 561 (yellow)
        - 488_2 (green)
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
    green = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.uint8)
    green[..., 1] = image2  # green
    return cyan, magenta, yellow, green


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


def generate_image_panel_whole_image(
    df: pd.DataFrame,
    input_file_parent_path: pathlib.Path,
    composite: bool = True,
    pseudo_color: bool = False,
) -> pd.DataFrame:
    """
    Generate a DataFrame containing composite images for a specific cell over time.


    Parameters
    ----------
    df : pd.DataFrame
        Image-based profile DataFrame.
        Each df should contain only one well_fov.
    composite : bool, optional
        Whether to generate composite images, by default True.
    pseudo_color : bool, optional
        Whether to generate pseudo-colored images, by default False.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: single_cell_id, well_fov, composite, time, dose.
    """
    # check that composite and pseudo_color are not both True or both False
    if composite == pseudo_color:
        raise ValueError("composite and pseudo_color cannot be both True or both False")

    df = df.sort_values("Metadata_Time")

    well_fov = df["Metadata_Well_FOV"].unique()[0]
    image1_path = pathlib.Path(
        input_file_parent_path / df["Metadata_Image_FileName_CL_488_1"].values[0]
    ).resolve(strict=True)
    image2_path = pathlib.Path(
        input_file_parent_path / df["Metadata_Image_FileName_CL_488_2"].values[0]
    ).resolve(strict=True)
    image3_path = pathlib.Path(
        input_file_parent_path / df["Metadata_Image_FileName_CL_561"].values[0]
    ).resolve(strict=True)
    image4_path = pathlib.Path(
        input_file_parent_path / df["Metadata_Image_FileName_DNA"].values[0]
    ).resolve(strict=True)
    if composite:
        output_dict = {
            "well_fov": [],
            "composite": [],
            "time": [],
            "dose": [],
        }
        composite_image = scale_image(
            make_composite_image(
                image1_path=image1_path,
                image2_path=image2_path,
                image3_path=image3_path,
                image4_path=image4_path,
            )
        )

        output_dict["well_fov"].append(df["Metadata_Well_FOV"].values[0])
        output_dict["composite"].append(composite_image)
        output_dict["time"].append(df["Metadata_Time"].values[0])
        output_dict["dose"].append(df["Metadata_dose"].values[0])
    if pseudo_color:
        output_dict = {
            "well_fov": [],
            "cyan": [],
            "magenta": [],
            "yellow": [],
            "green": [],
            "time": [],
            "dose": [],
        }
        cyan, magenta, yellow, green = make_pseudo_color_image(
            image1_path=image1_path,
            image2_path=image2_path,
            image3_path=image3_path,
            image4_path=image4_path,
        )
        cyan = scale_image(Image.fromarray(cyan))
        magenta = scale_image(Image.fromarray(magenta))
        yellow = scale_image(Image.fromarray(yellow))
        green = scale_image(Image.fromarray(green))

        enhancer = ImageEnhance.Contrast(cyan)
        cyan = enhancer.enhance(25)  # Increase contrast
        enhancer = ImageEnhance.Contrast(magenta)
        magenta = enhancer.enhance(25)  # Increase contrast
        enhancer = ImageEnhance.Contrast(yellow)
        yellow = enhancer.enhance(25)  # Increase contrast
        enhancer = ImageEnhance.Contrast(green)
        green = enhancer.enhance(25)  # Increase contrast

        output_dict["well_fov"].append(df["Metadata_Well_FOV"].values[0])
        output_dict["cyan"].append(cyan)
        output_dict["magenta"].append(magenta)
        output_dict["yellow"].append(yellow)
        output_dict["green"].append(green)
        output_dict["time"].append(df["Metadata_Time"].values[0])
        output_dict["dose"].append(df["Metadata_dose"].values[0])
    output_df = pd.DataFrame(output_dict)
    return output_df


# ## Load data and get pseudo colored images

# In[4]:


single_cell_profiles = pathlib.Path(
    "../../../data/CP_scDINO_features/combined_CP_scDINO_norm_fs.parquet"
).resolve(strict=True)
df = pd.read_parquet(single_cell_profiles)
df["Metadata_Well_FOV"] = df["Metadata_Well"] + "_F" + df["Metadata_FOV"]
df["Metadata_Time"] = df["Metadata_Time"].astype(float).astype(int)
df["Metadata_Time"].sort_values()
df.head()


# In[ ]:


list_of_dfs = []
# check if both are false or true, only one can be true
if not (composite ^ pseudo_color):
    raise ValueError("Either composite or pseudo_color must be True, but not both.")
total = 0
written = 0
existing = 0
for well_fov in tqdm(
    df["Metadata_Well_FOV"].unique(),
    desc="Processing well_fov",
    unit=" well_fov",
    leave=False,
):
    tmp_df = df.loc[(df["Metadata_Well_FOV"] == well_fov)]
    input_file_parent_path = pathlib.Path(
        "/home/lippincm/4TB_A/live_cell_timelapse_apoptosis/"
        "2.cellprofiler_ic_processing/illum_directory/timelapse/"
        f"20231017ChromaLive_6hr_4ch_MaxIP_{well_fov}"
    ).resolve(strict=True)
    for timepoint in tqdm(
        tmp_df["Metadata_Time"].unique(),
        desc="Processing timepoint",
        unit=" timepoint",
        leave=False,
    ):

        tmp_time_df = tmp_df.loc[tmp_df["Metadata_Time"] == timepoint].copy()
        tmp_time_df = tmp_time_df.drop_duplicates(
            subset=["Metadata_Well_FOV"]
        )  # Remove inplace=True
        output_df = generate_image_panel_whole_image(
            df=tmp_time_df,
            input_file_parent_path=input_file_parent_path,
            composite=composite,
            pseudo_color=pseudo_color,
        )
        if composite:
            image = output_df["composite"][0]
            # save path
            output_save_path = pathlib.Path(
                f"../data/whole_image_composite_images/{well_fov}_{timepoint:03d}_composite.png"
            )
            output_save_path.parent.mkdir(parents=True, exist_ok=True)
            total += 1
            if not output_save_path.exists():
                image.save(output_save_path)
                written += 1
            else:
                existsing += 1
        if pseudo_color:
            cyan, magenta, yellow, green = (
                output_df["cyan"][0],
                output_df["magenta"][0],
                output_df["yellow"][0],
                output_df["green"][0],
            )
            # save path
            output_save_path = pathlib.Path(
                f"../data/whole_image_pseudocolor_images/{well_fov}_{timepoint:03d}_cyan.png"
            )
            total += 4
            output_save_path.parent.mkdir(parents=True, exist_ok=True)
            if not output_save_path.exists():
                cyan.save(output_save_path)
                written += 1
            else:
                existsing += 1
            output_save_path = pathlib.Path(
                f"../data/whole_image_pseudocolor_images/{well_fov}_{timepoint:03d}_magenta.png"
            )
            if not output_save_path.exists():
                magenta.save(output_save_path)
                written += 1
            else:
                existsing += 1
            output_save_path = pathlib.Path(
                f"../data/whole_image_pseudocolor_images/{well_fov}_{timepoint:03d}_yellow.png"
            )
            if not output_save_path.exists():
                yellow.save(output_save_path)
                written += 1
            else:
                existsing += 1
            output_save_path = pathlib.Path(
                f"../data/whole_image_pseudocolor_images/{well_fov}_{timepoint:03d}_green.png"
            )
            if not output_save_path.exists():
                green.save(output_save_path)
                written += 1
            else:
                existsing += 1
print(f"Total images processed: {total}")
print(f"Total images written: {written}")
print(f"Total images already existing: {existsing}")
