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


def get_ml_fpath():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    ml_fpath = os.path.join(
        parent_dir, "ml", "models"
    )

    if not os.path.exists(ml_fpath):
        parent_dir = os.path.dirname(parent_dir)

    return os.path.join(
            parent_dir, "ml"
        )


def get_class_fpath(classes_fname):

    ml_dir = get_ml_fpath()

    classes_fpath = os.path.join(
        ml_dir, "models","labels",
        classes_fname
    )

    return classes_fpath