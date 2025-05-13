import numpy as np
import optuna
import torch
import torch.optim as optim
from build_custom_model import build_model_custom
from parameters import Parameters
from training import train_n_validate


def objective_model_optimizer(
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    trial: object = optuna.create_study,
    params: Parameters = False,
    metric: str = "loss",
    return_info: bool = False,
) -> str | int:
    """Optimizes the hyperparameter based on search space defined
    returns metrics of how well a given model is doing this is a helper
    function for the optuna optimizer

        Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        DataLoader for train data integration to pytorch
    valid_loader : torch.utils.data.DataLoader
        DataLoader for validation data integration to pytorch
    trial : trial from optuna
        hyperparameter optimization trial from optuna
    params : optuna.create_study
        Dataclass containing constants and parameter spaces
    metric : str
        metric to be tracked for model optimization
        valid options: 'accuracy' or 'loss'
    return_info : bool, optional
        the option to return more than one metric
        this is best to be False inside of the 'study.optimize' function
        as this function requires only one output metric, by default False



    Returns
    -------
    str | int
        str: returns printed statements of metrics
        when return_info=True
        int: returns metric specified in metric arg
        when return_info=False

    Raises
    ------
    optuna.exceptions.TrialPruned
        exception is raised if/when a trial is pruned due to poor intermediate values
    Exception
        raised when metric value is nor 'accuracy' or 'loss'
    """

    # calling model function
    model = build_model_custom(trial, params)

    # param dictionary for optimization
    optimization_params = {
        "learning_rate": trial.suggest_float(
            "learning_rate", params.LEARNING_RATE_MIN, params.LEARNING_RATE_MAX
        ),
        "optimizer": trial.suggest_categorical("optimizer", params.OPTIMIZER_LIST),
    }

    # param optimizer pick
    optimizer = getattr(optim, optimization_params["optimizer"])(
        model.parameters(), lr=optimization_params["learning_rate"]
    )
    # loss function
    criterion = torch.nn.MSELoss()

    # send model to device(cuda)
    model = model.to(params.DEVICE)
    criterion = criterion.to(params.DEVICE)

    # train set accuracy and loss
    train_loss = []

    # validation set accuracy and loss
    valid_loss = []

    # total number of data to pass through
    total_step = len(train_loader)
    total_step_val = len(valid_loader)

    for epoch in range(params.OPTIM_EPOCHS):
        (
            model,
            optimizer,
            criterion,
            train_loss,
            valid_loss,
        ) = train_n_validate(
            model,
            optimizer,
            criterion,
            train_loss,
            valid_loss,
            total_step,
            total_step_val,
            params,
            train_loader,
            valid_loader,
        )

        trial.report(np.mean(valid_loss), epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # for hyperparameter optimization we must return a single value
    return np.mean(valid_loss)
