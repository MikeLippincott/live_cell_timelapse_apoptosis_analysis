#!/usr/bin/env python
# coding: utf-8

# This notebook performs a linear model regression on timelapse data to understand the temporal contribution to each morphology feature.

# ## The linear model will be constructed as follows:
# ### $Y_{feature} = \beta_t X_t + \beta_{cell count} X_{cell count} + \beta_{Stuarosporine \space dose} X_{Stuarosporine \space dose} + \beta_0$

# In[ ]:


import pathlib
import warnings

import joblib
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
import tqdm

warnings.filterwarnings("ignore")


# In[2]:


def fit_linear_model(
    X: np.ndarray,
    y: np.ndarray,
    feature: str,
) -> statsmodels.regression.linear_model.RegressionResultsWrapper:
    """
    Fit a linear model to the data and save the model to a file.

    Parameters
    ----------
    X : np.ndarray
        The input data to fit on.
    y : np.ndarray
        The target data to fit on.
    feature : str
        The feature name that is used from y to fit the model.
    shuffle : bool
        Whether to shuffle the data before fitting the model.

    Returns
    -------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted model.
    """
    # Ensure X and y are numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    # Drop rows with missing values
    X = X.dropna()
    y = y.loc[X.index]

    # Select only numeric columns in X
    X = X.select_dtypes(include=[np.number])

    # Add a constant for the intercept
    X = sm.add_constant(X)

    # Fit the model
    model = sm.OLS(y, X).fit()
    # write the model to a file joblib
    joblib_path = pathlib.Path(f"linear_models/lm_{feature}.joblib").resolve()
    joblib_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, joblib_path)

    return model


# In[ ]:


agg_profile_file_path = pathlib.Path(
    "../../data/CP_scDINO_features/combined_CP_scDINO_norm_fs_aggregated.parquet"
).resolve(strict=True)

all_features_beta_df_path = pathlib.Path(
    "../results/all_features_beta_df.parquet"
).resolve()
all_features_beta_df_path.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(agg_profile_file_path)
df.head()


# In[4]:


# get the metadata features
metadata_columns = [x for x in df.columns if "Metadata" in x]
# get the features
feature_columns = [x for x in df.columns if "Metadata" not in x]
time_column = "Metadata_Time"
single_cells_count_column = "Metadata_number_of_singlecells"
dose_column = "Metadata_dose"


# In[5]:


coefficient_names = {
    "beta": [],
    "p_value": [],
    "variate": [],
    "r2": [],
    "feature": [],
}


# In[6]:


X = df[[time_column, single_cells_count_column, dose_column]]
for feature in tqdm.tqdm(feature_columns):
    y = df[feature]
    model = fit_linear_model(
        X,
        y,
        feature,
    )
    # get the model coefficients and p-values
    for variate in model.params.keys():
        coefficient_names["beta"].append(model.params[variate])
        coefficient_names["variate"].append(variate)
    for pval in model.pvalues.keys():
        coefficient_names["p_value"].append(model.pvalues[pval])
        coefficient_names["r2"].append(model.rsquared)
        coefficient_names["feature"].append(feature)


# In[7]:


all_features_beta_df = pd.DataFrame.from_dict(coefficient_names)
# remove any "Metadata_" string from the feature names
all_features_beta_df["variate"] = all_features_beta_df["variate"].str.replace(
    "Metadata_", ""
)
all_features_beta_df["variate"] = all_features_beta_df["variate"].str.replace(
    "number_of_singlecells", "Cell count"
)
# save the linear model coefficients to a file
all_features_beta_df.head()


# Extract feature information and save the dataframe

# In[8]:


# split the df into two dfs, one with CP and with scDINO
cp_df = all_features_beta_df[all_features_beta_df["feature"].str.contains("CP")]

cp_df = cp_df.copy()
cp_df[
    [
        "Compartment",
        "Feature_type",
        "Measurement",
        "Channel",
        "extra1",
        "extra2",
        "extra3",
        "extra4",
        "extra5",
        "extra6",
    ]
] = cp_df["feature"].str.split("_", expand=True)
cp_df["featurizer"] = "CP"

cp_df.loc[cp_df["Feature_type"].str.contains("AreaShape"), "Channel"] = "None"

cp_df.loc[cp_df["Channel"].str.contains("CL", na=False), "Channel"] = (
    cp_df["Channel"].fillna("") + "_" + cp_df["extra1"].fillna("")
)
cp_df.loc[cp_df["Channel"].str.contains("488", na=False), "Channel"] = (
    cp_df["Channel"].fillna("") + "_" + cp_df["extra2"].fillna("")
)

cp_df.loc[cp_df["extra1"].str.contains("CL", na=False), "extra1"] = (
    cp_df["extra1"].fillna("") + "_" + cp_df["extra2"].fillna("")
)
cp_df.loc[cp_df["extra1"].str.contains("488", na=False), "extra1"] = (
    cp_df["extra1"].fillna("") + "_" + cp_df["extra3"].fillna("")
)

cp_df.loc[cp_df["extra2"].str.contains("CL", na=False), "extra2"] = (
    cp_df["extra2"].fillna("") + "_" + cp_df["extra3"].fillna("")
)
cp_df.loc[cp_df["extra2"].str.contains("488", na=False), "extra2"] = (
    cp_df["extra2"].fillna("") + "_" + cp_df["extra4"].fillna("")
)

cp_df.loc[cp_df["extra3"].str.contains("CL", na=False), "extra3"] = (
    cp_df["extra3"].fillna("") + "_" + cp_df["extra4"].fillna("")
)
cp_df.loc[cp_df["extra3"].str.contains("488", na=False), "extra3"] = (
    cp_df["extra3"].fillna("") + "_" + cp_df["extra5"].fillna("")
)

cp_df.loc[cp_df["extra4"].str.contains("CL", na=False), "extra4"] = (
    cp_df["extra4"].fillna("") + "_" + cp_df["extra5"].fillna("")
)
cp_df.loc[cp_df["extra4"].str.contains("488", na=False), "extra4"] = (
    cp_df["extra4"].fillna("") + "_" + cp_df["extra6"].fillna("")
)

cp_df.rename(columns={"extra1": "Channel2"}, inplace=True)
cp_df.drop(
    columns=["Measurement", "extra2", "extra3", "extra4", "extra5", "extra6"],
    inplace=True,
)
# make channel2 None if feature is not correlation
cp_df.loc[~cp_df["Feature_type"].str.contains("Correlation"), "Channel2"] = "None"


# In[9]:


scdino_df = all_features_beta_df[all_features_beta_df["feature"].str.contains("scDINO")]
scdino_df = scdino_df.copy()
scdino_df["feature"] = scdino_df["feature"].str.replace("channel_", "")
scdino_df["feature"] = scdino_df["feature"].str.replace("channel", "")

scdino_df = scdino_df.copy()
scdino_df[["Channel", "remove", "feature", "feature_number", "featurizer"]] = scdino_df[
    "feature"
].str.split("_", expand=True)
scdino_df.drop(columns=["remove"], inplace=True)
scdino_df[["Compartment", "Feature_type", "Measurement"]] = "scDINO"


# In[10]:


final_df = pd.concat([cp_df, scdino_df], axis=0)
final_df["Channel"] = final_df["Channel"].str.replace("Adjacent", "None")

final_df["Channel"] = final_df["Channel"].str.replace("CL_488_1", "488-1")
final_df["Channel"] = final_df["Channel"].str.replace("CL_488_2", "488-2")
final_df["Channel"] = final_df["Channel"].str.replace("CL_561", "561")
final_df["Channel"] = final_df["Channel"].str.replace("488-1", "CL 488-1")
final_df["Channel"] = final_df["Channel"].str.replace("488-2", "CL 488-2")
final_df["Channel"] = final_df["Channel"].str.replace("561", "CL 561")
final_df["Channel"].unique()


# In[11]:


# save the final df to a file
final_df.to_parquet(all_features_beta_df_path, index=False)
final_df.head()
