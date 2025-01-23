from fastapi import (APIRouter, HTTPException,
                     File, UploadFile)
from services.predict import DeepLearningModelHandlerScore as dl_model
from services.validate import validate_image_file
from ml.models.prediction import PredictionResponse, ImageRequest
import io
from PIL import Image

from io import BytesIO
import base64


router = APIRouter()

## Change this portion for other types of models
## Add the correct type hinting when completed
def get_prediction(data_point):
    return dl_model.predict(data_point)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    name="predict:post-prediction-json",
)
async def predict(request: ImageRequest):
    try:
        # Decode the Base64-encoded image
        image_data = base64.b64decode(request.image)
        image = Image.open(BytesIO(image_data))  # Load image into PIL for further processing
        response = get_prediction(image)

        return response
    except base64.binascii.Error as e:
        raise HTTPException(status_code=400, detail="Invalid Base64-encoded image data")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/predict-form-data",
    response_model=PredictionResponse,
    name="predict:get-prediction",
)
async def predict_form_data(image: UploadFile = File(...)):

    if not image:
        raise HTTPException(status_code=404, 
                            detail="'image' argument invalid!")

    validate_image_file(image)

    try:
        contents = await image.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        response = get_prediction(image)
        return response

    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Exception: {err}")
