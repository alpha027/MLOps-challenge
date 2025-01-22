import numpy as np

from pydantic import BaseModel
from fastapi import  UploadFile



# Set the model to evaluation mode

ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png"]

class ImageInput(BaseModel):
    image: UploadFile

class PredictionResponse(BaseModel):
    response: str

class MachineLearningResponse(BaseModel):
    prediction: float
