import toml
from parameters import Parameters


def parameter_set(params: Parameters, config: toml) -> object:
    """reset parameter class defaults by updating from config

    Parameters
    ----------
    params : Parameters
        param class holding parameter information
    config: toml
        config class

    Returns
    -------
    object
        param class holding updated parameter information
    """
    params.DATA_SUBSET_OPTION = config["MACHINE_LEARNING_PARAMS"]["DATA_SUBSET_OPTION"]
    params.DATA_SUBSET_NUMBER = int(
        config["MACHINE_LEARNING_PARAMS"]["DATA_SUBSET_NUMBER"]
    )
    params.BATCH_SIZE = int(config["MACHINE_LEARNING_PARAMS"]["BATCH_SIZE"])

    params.OPTIM_EPOCHS = int(config["MACHINE_LEARNING_PARAMS"]["OPTIM_EPOCHS"])
    params.N_TRIALS = int(config["MACHINE_LEARNING_PARAMS"]["N_TRIALS"])
    params.TRAIN_EPOCHS = int(config["MACHINE_LEARNING_PARAMS"]["TRAIN_EPOCHS"])
    params.MIN_LAYERS = int(config["MACHINE_LEARNING_PARAMS"]["MIN_LAYERS"])
    params.MAX_LAYERS = int(config["MACHINE_LEARNING_PARAMS"]["MAX_LAYERS"])
    params.LAYER_UNITS_MIN = int(config["MACHINE_LEARNING_PARAMS"]["LAYER_UNITS_MIN"])
    params.LAYER_UNITS_MAX = int(config["MACHINE_LEARNING_PARAMS"]["LAYER_UNITS_MAX"])
    params.DROPOUT_MIN = float(config["MACHINE_LEARNING_PARAMS"]["DROPOUT_MIN"])
    params.DROPOUT_MAX = float(config["MACHINE_LEARNING_PARAMS"]["DROPOUT_MAX"])
    params.DROP_OUT_STEP = float(config["MACHINE_LEARNING_PARAMS"]["DROP_OUT_STEP"])
    params.LEARNING_RATE_MIN = float(
        config["MACHINE_LEARNING_PARAMS"]["LEARNING_RATE_MIN"]
    )
    params.LEARNING_RATE_MAX = float(
        config["MACHINE_LEARNING_PARAMS"]["LEARNING_RATE_MAX"]
    )
    params.OPTIMIZER_LIST = config["MACHINE_LEARNING_PARAMS"]["OPTIMIZER_LIST"]
    params.METRIC = config["MACHINE_LEARNING_PARAMS"]["METRIC"]
    params.DIRECTION = config["MACHINE_LEARNING_PARAMS"]["DIRECTION"]
    params.SHUFFLE = bool(config["MACHINE_LEARNING_PARAMS"]["SHUFFLE"])
    return params
