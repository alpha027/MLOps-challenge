from fastapi import (APIRouter, HTTPException,
                     File, UploadFile)
from services.predict import DeepLearningModelHandlerScore as dl_model
from services.validate import validate_image
from models.prediction import (
    PredictionResponse,
)
from fastapi.responses import JSONResponse
import io
from PIL import Image


router = APIRouter()

## Change this portion for other types of models
## Add the correct type hinting when completed
def get_prediction(data_point):
    return dl_model.predict(data_point)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    name="predict:get-prediction",
)
async def predict(image: UploadFile = File(...)):

    if not image:
        raise HTTPException(status_code=404, 
                            detail="'image' argument invalid!")

    validate_image(image)

    try:
        contents = await image.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        value = get_prediction(image)

    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Exception: {err}")

    #return MachineLearningResponse(prediction=prediction)
    return JSONResponse(value)
