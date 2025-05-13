from typing import Tuple

import torch
from build_custom_model import build_model_custom
from parameters import Parameters


def train_n_validate(
    model: build_model_custom,
    optimizer: str,
    criterion: torch.nn,
    train_loss: list,
    valid_loss: list,
    total_step: int,
    total_step_val: int,
    params: Parameters,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
) -> Tuple[build_model_custom, str, object, list, list, list, list, int, int, int, int]:
    """This function trains and validates a machine learning neural network
    the output is used as feedback for optuna hyper parameter optimization


    Parameters
    ----------
    model : build_model_custom
        initialized class containing model
    optimizer : str
        optimizer type
    criterion : nn
        criterion function to be used to calculate loss
    train_loss : list
        list for adding training loss values
    valid_loss : list
        list for adding validation loss values
    total_step : int
        the length of the number of data points in training dataset
    total_step_val : int
        the length of the number of data points in validation dataset
    params : Parameters
        Dataclass containing constants and parameter spaces
    train_loader : torch.utils.data.DataLoader
        DataLoader for train data integration to pytorch
    valid_loader : torch.utils.data.DataLoader
        DataLoader for validation data integration to pytorch


    Returns
    -------
    Tuple[build_model_custom, str, object, list, list, list, list, int, int, int, int]
        model : build_model_custom
            initialized class containing model
        optimizer : str
            optimizer type
        criterion : nn
            criterion function to be used to calculate loss
        train_loss : list
            list for adding training loss values
        valid_loss : list
            list for adding validation loss values
    """

    # TRAINING
    model.train()
    running_loss = 0

    for _, (X_train_batch, y_train_batch) in enumerate(train_loader):

        X_train_batch, y_train_batch = X_train_batch.to(
            params.DEVICE
        ), y_train_batch.to(params.DEVICE)
        optimizer.zero_grad()
        output = model(X_train_batch)
        loss = criterion(output, y_train_batch)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model weights
        running_loss += loss.item()

    if total_step > 0:
        train_loss.append(running_loss / total_step)  # Average loss
    else:
        train_loss.append(0)  # Average loss

    # VALIDATION
    batch_loss = 0
    with torch.no_grad():
        model.eval()
        for _, (X_valid_batch, y_valid_batch) in enumerate(valid_loader):
            X_valid_batch, y_valid_batch = X_valid_batch.to(
                params.DEVICE
            ), y_valid_batch.to(params.DEVICE)
            output = model(X_valid_batch)
            loss = criterion(output, y_valid_batch)
            batch_loss += loss.item()

    if total_step_val > 0:
        valid_loss.append(batch_loss / total_step_val)  # Average validation loss
    else:
        valid_loss.append(0)

    return (
        model,
        optimizer,
        criterion,
        train_loss,
        valid_loss,
    )
