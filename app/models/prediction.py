import numpy as np

from pydantic import BaseModel, validator
from fastapi import FastAPI, File, UploadFile, HTTPException



# Set the model to evaluation mode

ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png"]

class ImageInput(BaseModel):
    image: UploadFile

    # @validator('content_type')
    # def validate_content_type(cls, value):
    #     if value not in ALLOWED_CONTENT_TYPES:
    #         raise ValueError(f"Invalid content type. Allowed types are: {', '.join(ALLOWED_CONTENT_TYPES)}")
    #     return value


class PredictionResponse(BaseModel):
    response: str

class MachineLearningResponse(BaseModel):
    prediction: float


class HealthResponse(BaseModel):
    status: bool


class MachineLearningDataInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float

    def get_np_array(self):
        return np.array(
            [
                [
                    self.feature1,
                    self.feature2,
                    self.feature3,
                    self.feature4,
                    self.feature5,
                ]
            ]
        )
