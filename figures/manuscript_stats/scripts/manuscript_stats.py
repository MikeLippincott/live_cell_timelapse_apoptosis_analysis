#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import pandas as pd

# In[2]:


norm_profile_path = pathlib.Path(
    "../../../data/CP_scDINO_features/combined_CP_scDINO_norm.parquet"
)
fs_profile_path = pathlib.Path(
    "../../../data/CP_scDINO_features/combined_CP_scDINO_norm_fs.parquet"
)
norm_df = pd.read_parquet(norm_profile_path)

df = pd.read_parquet(fs_profile_path)
df.insert(
    0,
    "Metadata_sc_ID",
    df["Metadata_Well"]
    + "_"
    + df["Metadata_FOV"].astype(str)
    + "_"
    + df["Metadata_track_id"].astype(str),
)


# In[3]:


norm_metadata_cols = [col for col in norm_df.columns if "Metadata" in col]
fs_metadata_cols = [col for col in df.columns if "Metadata" in col]
norm_features_df = norm_df.drop(columns=norm_metadata_cols)
fs_features_df = df.drop(columns=fs_metadata_cols)


# In[4]:


norm_scDINO_features = [col for col in norm_features_df.columns if "scDINO" in col]
norm_CP_features = [col for col in norm_features_df.columns if "scDINO" not in col]
fs_scDINO_features = [col for col in fs_features_df.columns if "scDINO" in col]
fs_CP_features = [col for col in fs_features_df.columns if "scDINO" not in col]


# In[5]:


unique_cell_observations = df["Metadata_sc_ID"].nunique()
total_cell_observations = df.shape[0]


# In[6]:


print(
    f"Total number of features in normalized data (combined): {norm_features_df.shape[1]}"
)
print(
    f"Total number of features in normalized data (scDINO only): {len(norm_scDINO_features)}"
)
print(
    f"Total number of features in normalized data (CellProfiler only): {len(norm_CP_features)}"
)
print(
    f"Total number of features in feature selected data (combined): {fs_features_df.shape[1]}"
)
print(
    f"Total number of features in feature selected data (scDINO only): {len(fs_scDINO_features)}"
)
print(
    f"Total number of features in feature selected data (CellProfiler only): {len(fs_CP_features)}"
)
print(f"Number of unique cells observed: {unique_cell_observations}")
print(f"Number of total cell observations {total_cell_observations}")
