import optuna
import torch
import torch.nn as nn
from parameters import Parameters


def build_model_custom(
    trial: optuna.study,
    params: Parameters,
) -> torch.nn.Sequential:
    """Generate a flexible pytorch Neural Network Model that allows for
    optuna hyperparameter optimization

    Parameters
    ----------
    trial : optuna.study
        an iteration of the optuna study object
    params : Parameters
        parameter dataclass object to import constants and params

    Returns
    -------
    torch.nn.Sequential
        this returns in a dict the architecture of the model with optimized parameters
    """

    # number of hidden layers
    # suggest.int takes into account the defined search space
    n_layers = trial.suggest_int("n_layers", params.MIN_LAYERS, params.MAX_LAYERS)

    #  layers will be added to this list and called upon later
    layers = []
    in_features = params.IN_FEATURES

    for i in range(n_layers):
        # the number of units within a hidden layer
        out_features = trial.suggest_int(
            "n_units_l{}".format(i), params.LAYER_UNITS_MIN, params.LAYER_UNITS_MAX
        )

        layers.append(nn.Linear(in_features, out_features))

        layers.append(nn.ReLU())

        # dropout rate
        p = trial.suggest_float(
            (f"dropout_{i}"), params.DROPOUT_MIN, params.DROPOUT_MAX
        )

        layers.append(nn.Dropout(p))
        in_features = out_features

    # final layer append
    layers.append(nn.Linear(in_features, params.OUT_FEATURES))
    # add layers to the model
    return nn.Sequential(*layers)
