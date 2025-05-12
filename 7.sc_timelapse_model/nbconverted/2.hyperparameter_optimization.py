#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import pathlib
import sys

import joblib
import numpy as np
import optuna
import pandas as pd
import toml
import torch
from optuna.samplers import RandomSampler

sys.path.append("../ML_utils/")

from create_optimized_model import optimized_model_create
from extract_best_trial import extract_best_trial_params
from objective_creation import objective_model_optimizer
from parameter_set import parameter_set
from parameters import Parameters

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


# read in the data
sc_file_path = pathlib.Path("../results/cleaned_sc_profile.parquet").resolve(
    strict=True
)
sc_endpoint_file_path = pathlib.Path(
    "../results/cleaned_endpoint_sc_profile.parquet"
).resolve(strict=True)

data_split_file_path = pathlib.Path("../results/data_splits.parquet").resolve(
    strict=True
)

sc_profile = pd.read_parquet(sc_file_path)
sc_endpoint_profile = pd.read_parquet(sc_endpoint_file_path)
data_split_df = pd.read_parquet(data_split_file_path)
print(f"sc_profile shape: {sc_profile.shape}")
print(f"sc_endpoint_profile shape: {sc_endpoint_profile.shape}")
print(f"data_split_df shape: {data_split_df.shape}")


# In[3]:


# merge the sc_profile and data_split_df
sc_profile = pd.concat(
    [
        sc_profile,
        data_split_df[["ground_truth", "data_split"]],
    ],
    axis=1,
)
sc_profile.rename(
    columns={
        "ground_truth": "Metadata_ground_truth",
        "data_split": "Metadata_data_split",
    },
    inplace=True,
)


# In[4]:


# keep only the last timepoint
sc_profile["Metadata_Time"] = sc_profile["Metadata_Time"].astype("float64")
sc_profile = sc_profile[
    sc_profile["Metadata_Time"] == sc_profile["Metadata_Time"].max()
]
# drop Na values
sc_profile.dropna(inplace=True)
print(f"sc_profile shape after dropping NaN: {sc_profile.shape}")
sc_endpoint_profile.dropna(inplace=True)
print(f"sc_endpoint_profile shape after dropping NaN: {sc_endpoint_profile.shape}")


# In[5]:


# hardcode the features that should exist in the y data
# this will be replaced in the future by an arg or config passed through
selected_y_features = ["Cells_Intensity_MeanIntensityEdge_AnnexinV"]
metadata_y_features = [x for x in sc_endpoint_profile.columns if "Metadata_" in x]
sc_endpoint_profile = sc_endpoint_profile[metadata_y_features + selected_y_features]


# In[6]:


train_gt_X = sc_profile.loc[
    (sc_profile["Metadata_data_split"] == "train")
    & (sc_profile["Metadata_ground_truth"] == True)
]
val_gt_X = sc_profile.loc[
    (sc_profile["Metadata_data_split"] == "val")
    & (sc_profile["Metadata_ground_truth"] == True)
]
test_gt_X = sc_profile.loc[
    (sc_profile["Metadata_data_split"] == "test")
    & (sc_profile["Metadata_ground_truth"] == True)
]
test_wo_gt_X = sc_profile.loc[
    (sc_profile["Metadata_data_split"] == "test")
    & (sc_profile["Metadata_ground_truth"] == False)
]
holdout_w_gt_X = sc_profile.loc[
    (sc_profile["Metadata_data_split"] == "well_holdout")
    & (sc_profile["Metadata_ground_truth"] == True)
]
holdout_wo_gt_X = sc_profile.loc[
    (sc_profile["Metadata_data_split"] == "well_holdout")
    & (sc_profile["Metadata_ground_truth"] == False)
]
print(f"train_gt_X shape: {train_gt_X.shape}")
print(f"val_gt_X shape: {val_gt_X.shape}")
print(f"test_gt_X shape: {test_gt_X.shape}")
print(f"test_wo_gt_X shape: {test_wo_gt_X.shape}")
print(f"holdout_w_gt_X shape: {holdout_w_gt_X.shape}")
print(f"holdout_wo_gt_X shape: {holdout_wo_gt_X.shape}")


# In[7]:


# now let us get the the Metadata_sc_unique_track_id for each of the data splits with gt
train_df_x_Metadata_sc_unique_track_id = train_gt_X[
    "Metadata_sc_unique_track_id"
].unique()
val_df_x_Metadata_sc_unique_track_id = val_gt_X["Metadata_sc_unique_track_id"].unique()
test_df_x_Metadata_sc_unique_track_id = test_gt_X[
    "Metadata_sc_unique_track_id"
].unique()
holdout_w_df_x_Metadata_sc_unique_track_id = holdout_w_gt_X[
    "Metadata_sc_unique_track_id"
].unique()
print(
    f"train_df_x_Metadata_sc_unique_track_id shape: {train_df_x_Metadata_sc_unique_track_id.shape}"
)
print(
    f"val_df_x_Metadata_sc_unique_track_id shape: {val_df_x_Metadata_sc_unique_track_id.shape}"
)
print(
    f"test_df_x_Metadata_sc_unique_track_id shape: {test_df_x_Metadata_sc_unique_track_id.shape}"
)
print(
    f"holdout_w_df_x_Metadata_sc_unique_track_id shape: {holdout_w_df_x_Metadata_sc_unique_track_id.shape}"
)
# assertions :) make sure that the unique track ids are not overlapping
assert set(train_df_x_Metadata_sc_unique_track_id).isdisjoint(
    set(val_df_x_Metadata_sc_unique_track_id)
), "train and val track ids are overlapping"
assert set(train_df_x_Metadata_sc_unique_track_id).isdisjoint(
    set(test_df_x_Metadata_sc_unique_track_id)
), "train and test track ids are overlapping"
assert set(train_df_x_Metadata_sc_unique_track_id).isdisjoint(
    set(holdout_w_df_x_Metadata_sc_unique_track_id)
), "train and holdout track ids are overlapping"
assert set(val_df_x_Metadata_sc_unique_track_id).isdisjoint(
    set(test_df_x_Metadata_sc_unique_track_id)
), "val and test track ids are overlapping"
assert set(val_df_x_Metadata_sc_unique_track_id).isdisjoint(
    set(holdout_w_df_x_Metadata_sc_unique_track_id)
), "val and holdout track ids are overlapping"
assert set(test_df_x_Metadata_sc_unique_track_id).isdisjoint(
    set(holdout_w_df_x_Metadata_sc_unique_track_id)
), "test and holdout track ids are overlapping"


# In[8]:


# find only the cell tracks that exist in the sc_profile
train_gt_y = sc_endpoint_profile.loc[
    sc_endpoint_profile["Metadata_sc_unique_track_id"].isin(
        train_df_x_Metadata_sc_unique_track_id
    )
].drop_duplicates("Metadata_sc_unique_track_id")
val_gt_y = sc_endpoint_profile.loc[
    sc_endpoint_profile["Metadata_sc_unique_track_id"].isin(
        val_df_x_Metadata_sc_unique_track_id
    )
].drop_duplicates("Metadata_sc_unique_track_id")
test_gt_y = sc_endpoint_profile.loc[
    sc_endpoint_profile["Metadata_sc_unique_track_id"].isin(
        test_df_x_Metadata_sc_unique_track_id
    )
].drop_duplicates("Metadata_sc_unique_track_id")
holdout_gt_y = sc_endpoint_profile.loc[
    sc_endpoint_profile["Metadata_sc_unique_track_id"].isin(
        holdout_w_df_x_Metadata_sc_unique_track_id
    )
].drop_duplicates("Metadata_sc_unique_track_id")

# find only cell tracks that exist in the endpoint profile
train_gt_X = train_gt_X.loc[
    train_gt_X["Metadata_sc_unique_track_id"].isin(
        train_gt_y["Metadata_sc_unique_track_id"]
    )
].drop_duplicates("Metadata_sc_unique_track_id")
val_gt_X = val_gt_X.loc[
    val_gt_X["Metadata_sc_unique_track_id"].isin(
        val_gt_y["Metadata_sc_unique_track_id"]
    )
].drop_duplicates("Metadata_sc_unique_track_id")
test_gt_X = test_gt_X.loc[
    test_gt_X["Metadata_sc_unique_track_id"].isin(
        test_gt_y["Metadata_sc_unique_track_id"]
    )
].drop_duplicates("Metadata_sc_unique_track_id")
holdout_w_gt_X = holdout_w_gt_X.loc[
    holdout_w_gt_X["Metadata_sc_unique_track_id"].isin(
        holdout_gt_y["Metadata_sc_unique_track_id"]
    )
].drop_duplicates("Metadata_sc_unique_track_id")

print(f"train_y_gt shape: {train_gt_y.shape}, train_gt_X shape: {train_gt_X.shape}")
print(f"val_y_gt shape: {val_gt_y.shape}, val_gt_X shape: {val_gt_X.shape}")
print(f"test_y_gt shape: {test_gt_y.shape}, test_gt_X shape: {test_gt_X.shape}")
print(
    f"holdout_y_gt shape: {holdout_gt_y.shape}, holdout_gt_X shape: {holdout_w_gt_X.shape}"
)
# assertions :) make sure that the number of unique samples are the same
assert (
    train_gt_X.shape[0] == train_gt_y.shape[0]
), "train gt X and y shapes are not the same"
assert val_gt_X.shape[0] == val_gt_y.shape[0], "val gt X and y shapes are not the same"
assert (
    test_gt_X.shape[0] == test_gt_y.shape[0]
), "test gt X and y shapes are not the same"
assert (
    holdout_w_gt_X.shape[0] == holdout_gt_y.shape[0]
), "holdout gt X and y shapes are not the same"


# In[9]:


# get metadata
metadata_X_cols = [x for x in train_gt_X.columns if "Metadata_" in x]
metadata_y_cols = [x for x in train_gt_y.columns if "Metadata_" in x]
train_gt_X_metadata = train_gt_X[metadata_X_cols]
train_gt_X.drop(columns=metadata_X_cols, inplace=True)
val_gt_X_metadata = val_gt_X[metadata_X_cols]
val_gt_X.drop(columns=metadata_X_cols, inplace=True)
test_gt_X_metadata = test_gt_X[metadata_X_cols]
test_gt_X.drop(columns=metadata_X_cols, inplace=True)
holdout_w_gt_X_metadata = holdout_w_gt_X[metadata_X_cols]
holdout_w_gt_X.drop(columns=metadata_X_cols, inplace=True)
train_gt_y_metadata = train_gt_y[metadata_y_cols]
train_gt_y.drop(columns=metadata_y_cols, inplace=True)
val_gt_y_metadata = val_gt_y[metadata_y_cols]
val_gt_y.drop(columns=metadata_y_cols, inplace=True)
test_gt_y_metadata = test_gt_y[metadata_y_cols]
test_gt_y.drop(columns=metadata_y_cols, inplace=True)
holdout_w_gt_y_metadata = holdout_gt_y[metadata_y_cols]
holdout_gt_y.drop(columns=metadata_y_cols, inplace=True)


# In[10]:


# shuffle the data
shuffled_train_gt_X = train_gt_X.copy()
for col in shuffled_train_gt_X.columns:
    if col.startswith("Metadata_"):
        continue
    shuffled_train_gt_X[col] = np.random.permutation(shuffled_train_gt_X[col].values)
shuffled_val_gt_X = val_gt_X.copy()
for col in shuffled_val_gt_X.columns:
    if col.startswith("Metadata_"):
        continue
    shuffled_val_gt_X[col] = np.random.permutation(shuffled_val_gt_X[col].values)
shuffled_test_gt_X = test_gt_X.copy()
for col in shuffled_test_gt_X.columns:
    if col.startswith("Metadata_"):
        continue
    shuffled_test_gt_X[col] = np.random.permutation(shuffled_test_gt_X[col].values)
shuffled_holdout_w_gt_X = holdout_w_gt_X.copy()
for col in shuffled_holdout_w_gt_X.columns:
    if col.startswith("Metadata_"):
        continue
    shuffled_holdout_w_gt_X[col] = np.random.permutation(
        shuffled_holdout_w_gt_X[col].values
    )


# In[11]:


# number of input features
n_features = train_gt_X.shape[1]
# number of output features
n_outputs = train_gt_y.shape[1]
# number of metadata features
n_metadata_features = train_gt_X_metadata.shape[1]

print(f"n_features: {n_features}")
print(f"n_outputs: {n_outputs}")
print(f"n_metadata_features: {n_metadata_features}")


# In[12]:


dict_of_train_tests = {
    "train": {
        "X": train_gt_X,
        "y": train_gt_y,
        "metadata": train_gt_X_metadata,
        "model_path": [],
    },
    "val": {
        "X": val_gt_X,
        "y": val_gt_y,
        "metadata": val_gt_X_metadata,
        "model_path": [],
    },
    "test": {
        "X": test_gt_X,
        "y": test_gt_y,
        "metadata": test_gt_X_metadata,
        "model_path": [],
    },
    "train_shuffled": {
        "X": shuffled_train_gt_X,
        "y": train_gt_y,
        "metadata": train_gt_X_metadata,
        "model_path": [],
    },
    "val_shuffled": {
        "X": shuffled_val_gt_X,
        "y": val_gt_y,
        "metadata": val_gt_X_metadata,
        "model_path": [],
    },
    "test_shuffled": {
        "X": shuffled_test_gt_X,
        "y": test_gt_y,
        "metadata": test_gt_X_metadata,
        "model_path": [],
    },
}


# In[13]:


params = Parameters()
ml_configs = toml.load("../ML_utils/regression_class_config.toml")
mlp_params = parameter_set(params, ml_configs)
mlp_params.IN_FEATURES = n_features
mlp_params.OUT_FEATURES = n_outputs


# In[14]:


print(train_gt_X.shape, train_gt_y.shape)
print(val_gt_X.shape, val_gt_y.shape)


# In[15]:


X_train = torch.tensor(train_gt_X.values, dtype=torch.float32)
y_train = torch.tensor(train_gt_y.values, dtype=torch.float32)
X_val = torch.tensor(val_gt_X.values, dtype=torch.float32)
y_val = torch.tensor(val_gt_y.values, dtype=torch.float32)


# In[16]:


# get the dtypes of the data
print(f"X_train dtypes: {X_train.dtype}")
print(f"y_train dtypes: {y_train.dtype}")
print(f"X_val dtypes: {X_val.dtype}")
print(f"y_val dtypes: {y_val.dtype}")


# In[17]:


# produce data objects for train, val and test datasets
train_data = torch.utils.data.TensorDataset(X_train, y_train)
val_data = torch.utils.data.TensorDataset(X_val, y_val)


# convert data class into a dataloader to be compatible with pytorch
train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=mlp_params.HYPERPARAMETER_BATCH_SIZE, shuffle=True
)
valid_loader = torch.utils.data.DataLoader(
    dataset=val_data, batch_size=mlp_params.HYPERPARAMETER_BATCH_SIZE, shuffle=False
)


# In[18]:


logger = logging.getLogger()
logger.setLevel(logging.INFO)

pathlib.Path("../logs").mkdir(parents=True, exist_ok=True)
# Create a file handler
file_handler = logging.FileHandler("../logs/optuna_log.txt")
file_handler.setLevel(logging.INFO)

# Create a formatter and set it to the handler
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)

# Optional: Set Optuna to use this logger
optuna.logging.set_verbosity(optuna.logging.INFO)
optuna.logging.enable_propagation()


# In[ ]:


# wrap the objective function inside of a lambda function to pass args...
objective_lambda_func = lambda trial: objective_model_optimizer(
    train_loader,
    valid_loader,
    trial=trial,
    params=params,
    metric=mlp_params.METRIC,
    return_info=False,
)
# Study is the object for model optimization
study = optuna.create_study(
    direction=f"{mlp_params.DIRECTION}",
    sampler=RandomSampler(),
    study_name="live_cell_AnnexinV_prediction",
)
# Here I apply the optimize function of the study to the objective function
# This optimizes each parameter specified to be optimized from the defined search space
study.optimize(objective_lambda_func, n_trials=mlp_params.N_TRIALS)
# Prints out the best trial's optimized parameters
objective_model_optimizer(
    train_loader,
    valid_loader,
    trial=study.best_trial,
    params=params,
    metric=mlp_params.METRIC,
    return_info=True,
)


# In[23]:


model_name = "Cells_Intensity_MeanIntensityEdge_AnnexinV"
param_dict = extract_best_trial_params(study.best_params, params, model_name=model_name)


untrained_model_archetecture_only = optimized_model_create(
    params=params,
    model_name=model_name,
)
# save the blank model architecture
model_path = f"../models/{model_name}.pt"
