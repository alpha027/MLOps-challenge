from typing import Callable

from fastapi import FastAPI

from core.config import MODEL_NAME


def preload_model():
    """
    In order to load model on memory to each worker
    """
    from services.predict import DeepLearningModelHandlerScore

    DeepLearningModelHandlerScore.get_model(MODEL_NAME)


def create_start_app_handler(app: FastAPI) -> Callable:
    def start_app() -> None:
        preload_model()

    return start_app
