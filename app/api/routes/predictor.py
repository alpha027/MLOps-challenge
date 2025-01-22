import json

import joblib
from fastapi import APIRouter, HTTPException
from fastapi import FastAPI, File, UploadFile

from loguru import logger

from core.config import INPUT_EXAMPLE
from services.predict import MachineLearningModelHandlerScore as mlmodel
from models.prediction import (
    HealthResponse,
    MachineLearningResponse,
    MachineLearningDataInput,
    PredictionResponse,
    ImageInput
)

from fastapi.responses import JSONResponse
from torchvision import transforms

import io
from PIL import Image

import torch

router = APIRouter()

#model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)#weights="IMAGENET1K_V1")
model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', weights="IMAGENET1K_V1")
model.eval()


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

## Change this portion for other types of models
## Add the correct type hinting when completed
def get_prediction(data_point):
    return mlmodel.predict(data_point)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    name="predict:get-data",
)
async def predict(image: UploadFile = File(...)):

    if not image:
        raise HTTPException(status_code=404, detail="'image' argument invalid!")
    # try:
    #     image_input = ImageInput(filename=image.filename, content_type=image.content_type)
    # except ValueError as e:
    #     raise HTTPException(status_code=400, detail=str(e))
    try:
        image_format = image.content_type.lower()
        print(image_format)
        contents = await image.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted_class = outputs.max(1)  # Get the index of the max log-probability

        logger.info(f"Predicted class: {predicted_class.item()}")

        # Convert to class label
        #class_label = CLASS_LABELS[predicted_class.item()]
        class_label = "cat" if predicted_class.item() in [281, 282, 283,284,285] else "not cat" #CLASS_LABELS[predicted_class.item()]

    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Exception:")

    #return MachineLearningResponse(prediction=prediction)
    return JSONResponse({"response": class_label})


@router.post(
    "/predictor",
    response_model=PredictionResponse,
    name="predict:get-data",
)
async def predict(image: UploadFile = File(...)):

    if not image:
        raise HTTPException(status_code=404, detail="'image' argument invalid!")
    # try:
    #     image_input = ImageInput(filename=image.filename, content_type=image.content_type)
    # except ValueError as e:
    #     raise HTTPException(status_code=400, detail=str(e))
    try:
        image_format = image.content_type.lower()

        contents = await image.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        value = get_prediction(image)
        
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Exception: {err}")

    #return MachineLearningResponse(prediction=prediction)
    return JSONResponse(value)


@router.get(
    "/health",
    response_model=HealthResponse,
    name="health:get-data",
)
async def health():
    is_health = False
    try:
        test_input = MachineLearningDataInput(
            **json.loads(open(INPUT_EXAMPLE, "r").read())
        )
        test_point = test_input.get_np_array()
        get_prediction(test_point)
        is_health = True
        return HealthResponse(status=is_health)
    except Exception:
        raise HTTPException(status_code=404, detail="Unhealthy")
