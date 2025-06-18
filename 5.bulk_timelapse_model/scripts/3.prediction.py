#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import joblib
import numpy as np
import pandas as pd
import pycytominer

# In[2]:


train_test_wells_path = pathlib.Path(
    "../data_splits/train_test_wells.parquet"
).resolve()

predictions_save_path = pathlib.Path(
    "../results/predicted_terminal_profiles_from_all_time_points.parquet"
).resolve()

profile_data_path = pathlib.Path(
    "../../data/CP_scDINO_features/combined_CP_scDINO_norm_fs_aggregated.parquet"
).resolve()
terminal_column_names = pathlib.Path("../results/terminal_columns.txt").resolve(
    strict=True
)
terminal_column_names = [
    line.strip() for line in terminal_column_names.read_text().splitlines()
]

data_split_df = pd.read_parquet(train_test_wells_path)
df = pd.read_parquet(profile_data_path)
metadata_cols = [cols for cols in df.columns if "Metadata" in cols]
features_cols = [cols for cols in df.columns if "Metadata" not in cols]
features_cols = features_cols
aggregate_df = pycytominer.aggregate(
    population_df=df,
    strata=["Metadata_Well", "Metadata_Time"],
    features=features_cols,
    operation="median",
)


metadata_df = df[metadata_cols]
metadata_df = metadata_df.drop_duplicates(subset=["Metadata_Well", "Metadata_Time"])
metadata_df = metadata_df.reset_index(drop=True)
aggregate_df = pd.merge(
    metadata_df, aggregate_df, on=["Metadata_Well", "Metadata_Time"]
)
print(aggregate_df.shape)
aggregate_df.head()


# In[11]:


models_path = pathlib.Path("../models/").resolve(strict=True)
models = pathlib.Path(models_path).glob("*.joblib")
models_dict = {
    "model_name": [],
    "model_path": [],
    "shuffled": [],
    "feature": [],
}

for model_path in models:
    print(model_path.name)
    # print(model_path.name.split("singlefeature")[1].strip(".joblib").strip("_"))
    models_dict["model_name"].append(model_path.name)
    models_dict["model_path"].append(model_path)
    models_dict["shuffled"].append(
        "shuffled" if "shuffled" in model_path.name else "not_shuffled"
    )
    models_dict["feature"].append(
        model_path.name.split("singlefeature")[1].strip(".joblib").strip("_")
        if "singlefeature" in model_path.name
        else "all_terminal_features"
    )


# In[12]:


# map the train/test wells to the aggregate data
aggregate_df["Metadata_data_split"] = aggregate_df["Metadata_Well"].map(
    data_split_df.set_index("Metadata_Well")["data_split"]
)
data_split = aggregate_df.pop("Metadata_data_split")
aggregate_df.insert(0, "Metadata_data_split", data_split)
aggregate_df["Metadata_Time"] = aggregate_df["Metadata_Time"].astype(float)
# drop NaN values in the terminal columns
aggregate_df = aggregate_df.dropna(subset="Metadata_data_split")
aggregate_df["Metadata_data_split"].unique()


# In[13]:


# if the data_split is train and the time is not 12 then set to non_trained_pair
# where 12 is the last time point
aggregate_df["Metadata_data_split"] = aggregate_df.apply(
    lambda x: (
        "non_trained_pair"
        if (x["Metadata_data_split"] == "train" and x["Metadata_Time"] != 12.0)
        else x["Metadata_data_split"]
    ),
    axis=1,
)


# In[14]:


metadata_columns = [x for x in aggregate_df.columns if "metadata" in x.lower()]
aggregate_features_df = aggregate_df.drop(columns=metadata_columns, errors="ignore")


# In[15]:


models_dict["model_name"]


# In[19]:


aggregate_df


# In[16]:


results_dict = {}
for i, model_name in enumerate(models_dict["feature"]):
    model = joblib.load(models_dict["model_path"][i])
    if models_dict["feature"][i] != "all_terminal_features":
        print(models_dict["feature"][i])
        predicted_df = pd.DataFrame(
            model.predict(aggregate_features_df),
            columns=[models_dict["feature"][i]],
        )
    else:
        print("all_terminal_features")
        predicted_df = pd.DataFrame(
            model.predict(aggregate_features_df),
            columns=terminal_column_names,
        )
    predicted_df[metadata_columns] = aggregate_df[metadata_columns]
    predicted_df["shuffled"] = models_dict["shuffled"][i]
    # drop nan value
    predicted_df = predicted_df.dropna()

    # check if a key for the feature already exists in results_dict
    if f"{models_dict['feature'][i]}" in results_dict:
        temporary_df = pd.concat(
            [results_dict[f"{models_dict['feature'][i]}"], predicted_df],
            ignore_index=True,
            sort=False,
        )
        results_dict[f"{models_dict['feature'][i]}"] = temporary_df
    else:
        results_dict[f"{models_dict['feature'][i]}"] = predicted_df

    print(results_dict[f"{models_dict['feature'][i]}"].shape)


# In[17]:


for model in results_dict.keys():
    save_path = pathlib.Path(f"../results/{model}.parquet").resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results_dict[model].to_parquet(save_path, index=False)


# In[ ]:
