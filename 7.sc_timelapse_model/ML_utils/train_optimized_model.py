import pathlib

import numpy as np
import torch
import torch.nn as nn
from create_optimized_model import optimized_model_create
from parameters import Parameters
from training import train_n_validate


def train_optimized_model(
    EPOCHS: int,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    params: Parameters,
    model_name: str,
    shuffle: bool = False,
) -> tuple[float, float, float, float, int, torch.nn.Sequential]:
    """This function trains the optimized model on the training dataset


    Parameters
    ----------
    EPOCHS : int
        The number of epochs to train the model for
    train_loader : torch.utils.data.DataLoader
        DataLoader for train data integration to pytorch
    valid_loader : torch.utils.data.DataLoader
        DataLoader for train data integration to pytorch
    params : Parameters
        Dataclass containing constants and parameter spaces
    model_name : str
        name of the model to be added to the save name
    shuffle : bool, optional
        Whether to shuffle the data, by default True

    Returns
    -------
    tuple[float, float, float, float, int, object]
        train_loss: float
        train_acc: float
        valid_loss: float
        valid_acc: float
        epochs_ran: int
        model: torch.nn.Sequential

    """

    model, parameter_dict = optimized_model_create(params, model_name)
    model = model.to(params.DEVICE)
    # criterion is the method in which we measure our loss
    # isn't defined as loss as it doesn't represent the loss value but the method

    criterion = nn.MSELoss()

    if shuffle:
        model_name = f"{model_name}_shuffle"
    else:
        pass

    optim_method = parameter_dict["optimizer"].strip("'")
    print(optim_method)

    optimizer = f'optim.{optim_method}(model.parameters(), lr={parameter_dict["learning_rate"]})'

    optimizer = eval(optimizer)

    early_stopping_patience = 15
    early_stopping_counter = 0

    train_acc = []
    train_loss = []

    valid_acc = []
    valid_loss = []

    total_step = len(train_loader)
    total_step_val = len(valid_loader)

    valid_loss_min = np.inf

    epochs_ran = []

    for epoch in range(EPOCHS):
        epochs_ran.append(epoch + 1)

        (
            model,
            optimizer,
            criterion,
            train_acc,
            train_loss,
            valid_acc,
            valid_loss,
            correct,
            total,
            correct_v,
            total_v,
        ) = train_n_validate(
            model,
            optimizer,
            criterion,
            train_acc,
            train_loss,
            valid_acc,
            valid_loss,
            total_step,
            total_step_val,
            params,
            train_loader,
            valid_loader,
        )

        if np.mean(valid_loss) <= valid_loss_min:

            save_state_path = pathlib.Path(
                "../trained_models/model_save_states/Regression/"
            )
            pathlib.Path(save_state_path).mkdir(parents=True, exist_ok=True)
            torch.save(
                model.state_dict(),
                f"{save_state_path}/Regression_{model_name}.pt",
            )

            print(
                f"Epoch {epoch + 0:01}: Validation loss decreased ({valid_loss_min:.6f} --> {np.mean(valid_loss):.6f}).  Saving model ..."
            )
            valid_loss_min = np.mean(valid_loss)
            early_stopping_counter = 0  # reset counter if validation loss decreases
        else:
            print(f"Epoch {epoch + 0:01}: Validation loss did not decrease")
            early_stopping_counter += 1

        if early_stopping_counter > early_stopping_patience:
            print("Early stopped at epoch :", epoch)
            break

        print(
            f"\t Train_Loss: {np.mean(train_loss):.4f} Val_Loss: {np.mean(valid_loss):.4f}  BEST VAL Loss: {valid_loss_min:.4f} \n"
        )
    return train_loss, train_acc, valid_loss, valid_acc, epochs_ran, model
