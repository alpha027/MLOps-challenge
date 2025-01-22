import os

from loguru import logger

from core.errors import PredictException, ModelLoadException
from core.config import MODEL_NAME, MODEL_PATH
from models.registry import MODEL_REGISTRY
from services.validate import validate_registry
import torch
import json


class DeepLearningModelHandlerScore(object):
    model = None
    registry_name = None
    classes = None

    @classmethod
    def predict(cls, input):

        if cls.registry_name is not None:

            transform = MODEL_REGISTRY[cls.registry_name].get("transform",
                                                               None)
            if transform is not None:
                input = transform(input).unsqueeze(0)

        with torch.no_grad():
            outputs = cls.model(input)
            _, predicted_class = outputs.max(1)

        result = cls.classes[predicted_class.item()] if cls.classes is not None else str(predicted_class.item())
        return {"response":result}

    @classmethod
    def get_model(cls, model_name):

        if cls.model is not None:
            return cls.model

        if model_name.lower() in list(MODEL_REGISTRY.keys()):
            logger.info(f"Loading model {model_name}")
            cls.registry_name = model_name.lower()
            cls.model = cls.load(MODEL_REGISTRY[model_name.lower()])
            classes_fname = MODEL_REGISTRY[model_name.lower()].get("class", None)
            if classes_fname is not None:
                cls.classes = cls.loadClasses(classes_fname)
            if cls.model:
                return cls.model
        else:
            message = f"Model {model_name} not found in registry!"
            logger.error(message)
            raise ModelLoadException(message)

    @staticmethod
    def loadClasses(classes_fname):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)

        classes_fname = os.path.join(
            parent_dir, "models","labels",
            classes_fname
        )

        return json.load(open(classes_fname))

    @staticmethod
    def load(load_parameters):

        validate_registry(load_parameters)
        loading_method = load_parameters.get("loading_method")
        if loading_method.get("hub", None) is not None:
            model = torch.hub.load(
                loading_method["hub"]["url"],
                loading_method["hub"]["model"],
                weights=loading_method["hub"]["weights"]
            )
        elif loading_method.get("file", None) is not None:
            # Implement the logic to load the model from a ".pth" file
            model = torch.load(loading_method["file"]["path"])
        else:
            message = f"Model loading method not found!"
            logger.error(message)
            raise ModelLoadException(message)
        model.eval()

        return model