import pathlib
from typing import Tuple

import torch
import torch.nn as nn
from parameters import Parameters


def test_optimized_model(
    model: torch.nn.Sequential,
    test_loader: torch.utils.data.DataLoader,
    params: Parameters,
    model_name: str,
    shuffle: bool = False,
) -> Tuple[list, list]:
    """test the trained model on test data

    Parameters
    ----------
    model : torch.nn.Sequential
        pytorch model to us
    test_loader : torch.utils.data.DataLoader
        DataLoader for test data integration to pytorch
    params : Parameters
        Dataclass containing constants and parameter spaces
    model_name : str
        name of the model to be used for loading

    Returns
    -------
    Tuple[list, list]
        y_pred_list: list of predicted values for Y data

        y_pred_prob_list: list of probabilities of
        those predicted values
    """
    if shuffle:
        model_name = f"{model_name}_shuffle"
    else:
        pass
    print(model_name)
    model = model.to(params.DEVICE)

    save_state_path = pathlib.Path("../trained_models/model_save_states/Regression/")
    model.load_state_dict(torch.load(f"{save_state_path}/{model_name}.pt"))

    y_pred_list = []
    Y_test_list = []

    with torch.no_grad():
        model.eval()
        for _, (X_test_batch, Y_test_batch) in enumerate(test_loader):
            X_test_batch = X_test_batch.type(torch.FloatTensor)
            X_test_batch = X_test_batch.to(params.DEVICE)
            Y_test_batch = Y_test_batch.type(torch.FloatTensor)
            Y_test_batch = Y_test_batch.to(params.DEVICE)
            # PREDICTION
            output = model(X_test_batch)
            y_pred_list.append(output.cpu().numpy())
            Y_test_list.append(Y_test_batch.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    return y_pred_list, Y_test_list
