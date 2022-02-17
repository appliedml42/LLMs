import shutil
import importlib
import torch
import torch.nn.functional as F

def get_model_class(cmd, copy=True):
    """
    Get the model class.
    :param cmd: Namespace object with parsed command line arguments.
    :return: Model class
    """
    model_module_str = ".".join(cmd.model.split(".")[:-1])
    model_class_str = cmd.model.split(".")[-1]

    model_module = importlib.import_module(model_module_str)
    if copy:
        module_fpath = model_module.__file__
        shutil.copy(module_fpath, cmd.experiment_dpath)
    model_class = getattr(model_module, model_class_str)

    return model_class


def get_dataset_class(cmd, copy=True):
    """
    Get the Dataset class.
    :param cmd: Namespace object with parsed command line arguments.
    :return: Dataset class
    """
    dataset_module_str = ".".join(cmd.dataset.split(".")[:-1])
    dataset_class_str = cmd.dataset.split(".")[-1]

    dataset_module = importlib.import_module(dataset_module_str)

    if copy:
        dataset_module_fpath = dataset_module.__file__
        shutil.copy(dataset_module_fpath, cmd.experiment_dpath)
    dataset_class = getattr(dataset_module, dataset_class_str)

    return dataset_class