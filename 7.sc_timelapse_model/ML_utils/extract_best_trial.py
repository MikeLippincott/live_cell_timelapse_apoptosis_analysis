import json
import pathlib

import optuna
from parameters import Parameters


def extract_best_trial_params(
    best_params: optuna.study, MLP_params: Parameters, model_name: str
) -> dict:
    """This function extracts the best parameters from the best trial.
    These extracted parameters will be used to train a new model.

    Parameters
    ----------
    best_params : optuna.study.best_params
        returns the best paramaters from the best study from optuna
    MLP_params : Parameters
        dataclass containing constants and parameter spaces
    model_name : str
        name of the model to be created

    Returns
    -------
    dict
        dictionary of all of the params for the best trial model
    """

    units = []
    dropout = []
    n_layers = best_params["n_layers"]
    optimizer = best_params["optimizer"]
    lr = best_params["learning_rate"]
    for i in range(best_params["n_layers"]):
        units.append(best_params[f"n_units_l{i}"])
        dropout.append(best_params[f"dropout_{i}"])
        param_dict = {
            "units": units,
            "dropout": dropout,
            "n_layers": n_layers,
            "optimizer": optimizer,
            "learning_rate": lr,
        }

    # write model architecture to file

    architecture_path = pathlib.Path("../trained_models/architectures/")
    pathlib.Path(architecture_path).mkdir(parents=True, exist_ok=True)
    with open(
        f"{architecture_path}/{model_name}.json",
        "w",
    ) as f:
        json.dump(param_dict, f, indent=4)
    f.close()

    return param_dict
