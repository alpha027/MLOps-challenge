import numpy as np

from pydantic import BaseModel
from fastapi import  UploadFile



# Set the model to evaluation mode

ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png"]

# Define the request model
class ImageRequest(BaseModel):
    image: str  # Base64-encoded string

class ImageInput(BaseModel):
    image: UploadFile

class PredictionResponse(BaseModel):
    response: str

class MachineLearningResponse(BaseModel):
    prediction: float
