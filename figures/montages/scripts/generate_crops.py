#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
from typing import Tuple

import pandas as pd
import tifffile

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False
if in_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# In[2]:


radius = 75  # pixels to expand around centroid for crop


# In[ ]:


def crop_image(
    image_input_path: pathlib.Path,
    image_output_path: pathlib.Path,
    x_center: int,
    y_center: int,
    radius: int = 50,
) -> Tuple[int, int, int]:
    """
    Crop a square region from a larger image centered at (x_center, y_center) with given radius.

    Parameters
    ----------
    image_input_path : pathlib.Path
        The image to crop from as a path
    image_output_path : pathlib.Path
        The path to save the cropped image
    x_center : int
        The x-coordinate of the center of the crop
    y_center : int
        The y-coordinate of the center of the crop
    radius : int, optional
        The radius of the crop. Note this is radius from the x,y center.
        Also not a true radius but you get the point, by default 50

    Returns
    -------
    Tuple[int, int, int]
        A tuple of (omitted_count, successful_count, total_count)
        This indicates whether the crop was successful or omitted due to being an edge case.
    """
    omitted_count, successful_count, total_count = 0, 0, 1
    image = tifffile.imread(image_input_path)

    image_crop = image[
        y_center - radius : y_center + radius, x_center - radius : x_center + radius
    ]

    # check if crop is an edge case
    # Where edge case is cells that are too close to the edge of the image to crop
    # This ensures that all crops are the same dimensions and can be used in the model
    if image_crop.shape[0] < radius * 2 or image_crop.shape[1] < radius * 2:
        omitted_count = 1
        return (omitted_count, successful_count, total_count)

    image_output_path.parent.mkdir(parents=True, exist_ok=True)

    tifffile.imwrite(image_output_path, image_crop)
    successful_count = 1
    return (omitted_count, successful_count, total_count)


# In[4]:


single_cell_profiles = pathlib.Path(
    "../../data/CP_scDINO_features/combined_CP_scDINO_norm_fs.parquet"
).resolve(strict=True)


# In[5]:


input_file_parent_path = pathlib.Path(
    "/home/lippincm/4TB_A/live_cell_timelapse_apoptosis/"
).resolve(strict=True)
output_crop_parent_path = pathlib.Path("../../data/single_cell_crops/").resolve()
output_crop_parent_path.mkdir(parents=True, exist_ok=True)


# In[6]:


df = pd.read_parquet(single_cell_profiles)
df["Metadata_Image_FileName_CL_488_1_crop"] = (
    str(output_crop_parent_path)
    + "/"
    + df["Metadata_Image_FileName_CL_488_1"].str.replace(".tif", "")
    + "_crop_"
    + df["Metadata_Nuclei_Number_Object_Number"].astype(str)
    + "_"
    + df.index.astype(str)
    + ".tiff"
)
df["Metadata_Image_FileName_CL_488_2_crop"] = (
    str(output_crop_parent_path)
    + "/"
    + df["Metadata_Image_FileName_CL_488_2"].str.replace(".tif", "")
    + "_crop_"
    + df["Metadata_Nuclei_Number_Object_Number"].astype(str)
    + "_"
    + df.index.astype(str)
    + ".tiff"
)
df["Metadata_Image_FileName_CL_561_crop"] = (
    str(output_crop_parent_path)
    + "/"
    + df["Metadata_Image_FileName_CL_561"].str.replace(".tif", "")
    + "_crop_"
    + df["Metadata_Nuclei_Number_Object_Number"].astype(str)
    + "_"
    + df.index.astype(str)
    + ".tiff"
)
df["Metadata_Image_FileName_DNA_crop"] = (
    str(output_crop_parent_path)
    + "/"
    + df["Metadata_Image_FileName_DNA"].str.replace(".tif", "")
    + "_crop_"
    + df["Metadata_Nuclei_Number_Object_Number"].astype(str)
    + "_"
    + df.index.astype(str)
    + ".tiff"
)
df["Metadata_parent_path"] = df["Metadata_Image_PathName_CL_488_2"].apply(
    lambda x: f"{input_file_parent_path}{x.split('live_cell_timelapse_apoptosis')[1]}"
)
df.to_parquet(single_cell_profiles, index=False)


# In[7]:


omitted_count, successful_count, total_count = 0, 0, 0
for index in tqdm(df.index, desc="Generating crops...", total=len(df)):

    o1, s1, t1 = crop_image(
        image_input_path=pathlib.Path(
            f"{df['Metadata_parent_path'][index]}/{df['Metadata_Image_FileName_CL_488_1'][index]}"
        ),
        image_output_path=pathlib.Path(
            f"{df['Metadata_Image_FileName_CL_488_1_crop'][index]}"
        ),
        x_center=df["Metadata_x"][index].astype(int),
        y_center=df["Metadata_y"][index].astype(int),
        radius=radius,
    )
    o2, s2, t2 = crop_image(
        image_input_path=pathlib.Path(
            f"{df['Metadata_parent_path'][index]}/{df['Metadata_Image_FileName_CL_488_2'][index]}"
        ),
        image_output_path=pathlib.Path(
            f"{df['Metadata_Image_FileName_CL_488_2_crop'][index]}"
        ),
        x_center=df["Metadata_x"][index].astype(int),
        y_center=df["Metadata_y"][index].astype(int),
        radius=radius,
    )
    o3, s3, t3 = crop_image(
        image_input_path=pathlib.Path(
            f"{df['Metadata_parent_path'][index]}/{df['Metadata_Image_FileName_DNA'][index]}"
        ),
        image_output_path=pathlib.Path(
            f"{df['Metadata_Image_FileName_DNA_crop'][index]}"
        ),
        x_center=df["Metadata_x"][index].astype(int),
        y_center=df["Metadata_y"][index].astype(int),
        radius=radius,
    )
    o4, s4, t4 = crop_image(
        image_input_path=pathlib.Path(
            f"{df['Metadata_parent_path'][index]}/{df['Metadata_Image_FileName_CL_561'][index]}"
        ),
        image_output_path=pathlib.Path(
            f"{df['Metadata_Image_FileName_CL_561_crop'][index]}"
        ),
        x_center=df["Metadata_x"][index].astype(int),
        y_center=df["Metadata_y"][index].astype(int),
        radius=radius,
    )
    omitted_count += o1 + o2 + o3 + o4
    successful_count += s1 + s2 + s3 + s4
    total_count += t1 + t2 + t3 + t4


# In[8]:


print(
    f"Omitted {omitted_count} crops. Successfully created {successful_count} crops out of {total_count} total crops."
)
