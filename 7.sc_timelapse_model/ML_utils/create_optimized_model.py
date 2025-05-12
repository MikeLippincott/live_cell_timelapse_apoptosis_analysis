import json
import pathlib

import torch
from parameters import Parameters


def optimized_model_create(
    # parameter_dict: dict,
    params: Parameters,
    model_name: str,
) -> torch.nn.Sequential:
    """creates the pytorch model architecture from the best trial
    from optuna hyperparameter optimization

    Parameters
    ----------
    # parameter_dict : dict
    #     dictionary of optimized model hyperparameters
    params : Parameters
        dataclass to store data of hyperparameters and parameters
    model_name : str
        name of the model to be created

    Returns
    -------
    torch.nn.Sequential
        this returns in a dict the architecture of the model with optimized parameters
    """
    # load in model architecture from saved model architecture

    architecture_path = pathlib.Path("../trained_models/architectures/")
    with open(
        f"{architecture_path}/{model_name}.json",
        "r",
    ) as f:
        parameter_dict = json.load(f)
    f.close()

    n_layers = parameter_dict["n_layers"]

    layers = []
    in_features = params.IN_FEATURES
    # loop through each layer
    for i in range(n_layers):
        # for each layer access the correct hyper paramter
        out_features = parameter_dict["units"][i]

        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(torch.nn.ReLU())
        p = parameter_dict["dropout"][i]
        layers.append(torch.nn.Dropout(p))
        in_features = out_features
    layers.append(torch.nn.Linear(in_features, params.OUT_FEATURES))
    # output new model to train and test
    return torch.nn.Sequential(*layers), parameter_dict
