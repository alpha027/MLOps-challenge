import torch
from loguru import logger
from core.errors import ModelLoadException
import os


def load_model_from_hub(url, model, weights):
    try:
        model = torch.hub.load(
            url,
            model,
            weights=weights
        )
        return model
    except Exception as e:
        message = f"Error loading model from hub: {e}"
        logger.error(message)
        raise ModelLoadException(message)


def load_model_from_file(fname):

    raise NotImplementedError


def get_class_fpath(classes_fname):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    classes_fpath = os.path.join(
        parent_dir, "models","labels",
        classes_fname
    )

    return classes_fpath