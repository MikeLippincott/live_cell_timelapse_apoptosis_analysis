#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import pathlib

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA

# In[ ]:


# set the arg parser
parser = argparse.ArgumentParser(description="UMAP on a matrix")

parser.add_argument("--data_mode", type=str, default="CP", help="data mode to use")

# get the args
args = parser.parse_args()

# set data mode to either "CP" or "scDINO" or "combined" or "terminal"
data_mode = args.data_mode


# In[3]:


# set the paths to the data
CP_fs_sc_profiles_path = pathlib.Path(
    "../../data/CP_feature_select/profiles/features_selected_profile.parquet"
).resolve(strict=True)
combined_profiles_path = pathlib.Path(
    "../../data/CP_scDINO_features/combined_CP_scDINO_norm_fs.parquet"
).resolve(strict=True)
scDINO_fs_profiles_path = pathlib.Path(
    "../../data/scDINO/CLS_features_annotated_normalized_feature_selected.parquet"
).resolve(strict=True)

CP_endpoint_profiles_path = pathlib.Path(
    "../../data/CP_feature_select/endpoints/features_selected_profile.parquet"
).resolve(strict=True)

output_path = pathlib.Path("../../data/PCA/").resolve()
output_path.mkdir(parents=True, exist_ok=True)


# In[4]:


if data_mode == "CP":
    # read the data
    profiles_df = pd.read_parquet(CP_fs_sc_profiles_path)
elif data_mode == "combined":
    # read the data
    profiles_df = pd.read_parquet(combined_profiles_path)
elif data_mode == "scDINO":
    # read the data
    profiles_df = pd.read_parquet(scDINO_fs_profiles_path)
elif data_mode == "terminal":
    # read the data
    profiles_df = pd.read_parquet(CP_endpoint_profiles_path)
else:
    raise ValueError("data_mode must be either 'CP' or 'scDINO' or 'combined'")
print(profiles_df.shape)
# show all columns
pd.set_option("display.max_columns", None)
profiles_df.head()


# In[5]:


features_df = profiles_df.copy()
metadata_columns = [x for x in features_df.columns if "Metadata_" in x]
features_df = features_df.drop(metadata_columns, axis=1)
features_df.dropna(axis=0, inplace=True)
# pc
# separate the metadata
metadata_df = profiles_df[metadata_columns]
# fit the pca model
pca_model = PCA(n_components=2)
pca_model.fit(features_df)
# get the pca embeddings
pca_embeddings = pca_model.transform(features_df)
# create a dataframe with the pca fit and the metadata
pca_df = pd.DataFrame(pca_embeddings, columns=["PCA0", "PCA1"])
# add the metadata to the dataframe
pca_df = pd.concat([pca_df, metadata_df], axis=1)
pca_df.dropna(axis=0, inplace=True)
pca_df


# In[6]:


output_file_path = pathlib.Path(
    output_path, f"PCA_2D_{data_mode}_features.parquet"
).resolve()
pca_df.to_parquet(output_file_path)
