#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import warnings
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor

# ## Import the data

# In[2]:


# load the training data
train_data_file_path = pathlib.Path("../data_splits/train.parquet").resolve(strict=True)
test_data_file_path = pathlib.Path("../data_splits/test.parquet").resolve(strict=True)
model_dir = pathlib.Path("../models/").resolve()
model_dir.mkdir(parents=True, exist_ok=True)
results_dir = pathlib.Path("../results/").resolve()
results_dir.mkdir(parents=True, exist_ok=True)
train_df = pd.read_parquet(train_data_file_path)
test_df = pd.read_parquet(test_data_file_path)
train_df.head()


# In[3]:


metadata_columns = [x for x in train_df.columns if "Metadata" in x]
terminal_columns = [
    x for x in train_df.columns if "Terminal" in x and "Metadata" not in x
]


def shuffle_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shuffle the data in the DataFrame.
    """
    df_shuffled = df.copy()
    for col in df_shuffled.columns:
        # permute the columns
        df_shuffled[col] = np.random.permutation(df_shuffled[col])
    return df_shuffled


def x_y_data_separator(
    df: pd.DataFrame,
    y_columns: list,
    metadata_columns: list,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Separate the data into X, y, and metadata.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to separate. ASSUMPTION:
            The metadata columns contain the string "Metadata" and the y columns contain the string "Terminal".
            The column names are passed in as lists.
    y_columns : list
        The y columns to separate.
    metadata_columns : list
        The metadata columns to separate.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Three DataFrames: X, y, and metadata.
    """
    metadata = df[metadata_columns]
    df.drop(columns=metadata_columns, inplace=True)
    X = df.drop(columns=y_columns)
    y = df[y_columns]
    return X, y, metadata


shuffled_train_df = train_df.copy()
shuffled_train_df = shuffle_data(shuffled_train_df)
shuffled_test_df = test_df.copy()
shuffled_test_df = shuffle_data(shuffled_test_df)

# split the data into train and test sets
# train
(train_X, train_y, train_metadata) = x_y_data_separator(
    df=train_df, y_columns=terminal_columns, metadata_columns=metadata_columns
)
(train_shuffled_X, train_shuffled_y, train_metadata_shuffled) = x_y_data_separator(
    df=shuffled_train_df, y_columns=terminal_columns, metadata_columns=metadata_columns
)

# test
(test_X, test_y, test_metadata) = x_y_data_separator(
    df=test_df, y_columns=terminal_columns, metadata_columns=metadata_columns
)
(test_shuffled_X, test_shuffled_y, test_metadata_shuffled) = x_y_data_separator(
    df=shuffled_test_df, y_columns=terminal_columns, metadata_columns=metadata_columns
)


# check the shape of the data
print(f"train_X shape: {train_X.shape}, train_y shape: {train_y.shape}")
print(
    f"train_shuffled_X shape: {train_shuffled_X.shape}, train_shuffled_y shape: {train_shuffled_y.shape}"
)

print(f"test_X shape: {test_X.shape}, test_y shape: {test_y.shape}")
print(
    f"test_shuffled_X shape: {test_shuffled_X.shape}, test_shuffled_y shape: {test_shuffled_y.shape}"
)


# In[4]:


model_features = [
    "Terminal_Cytoplasm_Intensity_MaxIntensity_AnnexinV",
    "Terminal_Cytoplasm_Intensity_IntegratedIntensity_AnnexinV",
]


# In[5]:


dict_of_train_tests = {
    "train": {
        "X": train_X,
        "y": train_y,
        "metadata": train_metadata,
    },
    "train_shuffled": {
        "X": train_shuffled_X,
        "y": train_shuffled_y,
        "metadata": train_metadata_shuffled,
    },
    "test": {
        "X": test_X,
        "y": test_y,
        "metadata": test_metadata,
    },
    "test_shuffled": {
        "X": test_shuffled_X,
        "y": test_shuffled_y,
        "metadata": test_metadata_shuffled,
    },
}


# ## Model training

# In[6]:


# Define cross-validation strategy
cv = KFold(n_splits=5, shuffle=True, random_state=0)  # 5-fold cross-validation
# elastic net parameters
elastic_net_params = {
    "alpha": [0.1, 1.0, 10.0, 100.0, 1000.0],  # Regularization strength
    "l1_ratio": [0.1, 0.25, 0.5],  # l1_ratio = 1.0 is Lasso
    "max_iter": 10000,  # Increase max_iter for convergence
}
elastic_net_all_terminal_features_model = MultiOutputRegressor(
    ElasticNetCV(
        alphas=elastic_net_params["alpha"],
        l1_ratio=elastic_net_params["l1_ratio"],
        cv=cv,
        random_state=0,
        max_iter=elastic_net_params["max_iter"],
    )
)

elastic_net_single_terminal_features_model = ElasticNetCV(
    alphas=elastic_net_params["alpha"],
    l1_ratio=elastic_net_params["l1_ratio"],
    cv=cv,
    random_state=0,
    max_iter=elastic_net_params["max_iter"],
)

# train the model
for train_test_key, train_test_data in tqdm.tqdm(dict_of_train_tests.items()):
    if "test" in train_test_key:
        print(f"Skipping {train_test_key} as it is a test set.")
        continue
    X = train_test_data["X"]
    y = train_test_data["y"]
    metadata = train_test_data["metadata"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        elastic_net_all_terminal_features_model.fit(X, y)

    # save the model
    model_path = (
        model_dir / f"{train_test_key}_elastic_net_model_all_terminal_features.joblib"
    )
    joblib.dump(elastic_net_all_terminal_features_model, model_path)
    dict_of_train_tests[train_test_key]["model_path"] = model_path

    for single_feature in model_features:
        # Fit the model with a single terminal feature
        y_single_feature = y[[single_feature]]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            elastic_net_single_terminal_features_model.fit(X, y_single_feature)

        # Save the model
        single_feature_model_path = (
            model_dir
            / f"{train_test_key}_elastic_net_model_singlefeature_{single_feature}.joblib"
        )
        joblib.dump(
            elastic_net_single_terminal_features_model, single_feature_model_path
        )
        dict_of_train_tests[train_test_key][
            f"model_path_{single_feature}"
        ] = single_feature_model_path


# In[7]:


# test the model
for train_test_key, train_test_data in tqdm.tqdm(dict_of_train_tests.items()):
    if "train" in train_test_key:
        print(f"Skipping {train_test_key} as it is a training set.")
        continue
    X = train_test_data["X"]
    y = train_test_data["y"]
    metadata = train_test_data["metadata"]
    if "shuffled" in train_test_key:
        model_path = dict_of_train_tests["train_shuffled"]["model_path"]
    else:
        model_path = dict_of_train_tests["train"]["model_path"]

    # load the model
    model = joblib.load(model_path)

    # make predictions
    y_pred = model.predict(X)

    alphas = model.estimators_[0].alpha_
    l1_ratios = model.estimators_[0].l1_ratio_
    print(f"Model parameters for {train_test_key}:")
    print(f"Alphas: {alphas}, L1 Ratios: {l1_ratios}")

    # calculate metrics
    metrics = {
        "explained_variance": explained_variance_score(y, y_pred),
        "mean_absolute_error": mean_absolute_error(y, y_pred),
        "mean_squared_error": mean_squared_error(y, y_pred),
        "r2_score": r2_score(y, y_pred),
    }


# In[8]:


terminal_columns_file_path = results_dir / "terminal_columns.txt"
with open(terminal_columns_file_path, "w") as f:
    for col in terminal_columns:
        f.write(f"{col}\n")
