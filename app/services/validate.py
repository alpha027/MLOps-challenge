from fastapi import HTTPException
from fastapi import UploadFile


def validate_image_file(image: UploadFile):
    try:
        if image.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(
                status_code=400, detail="Image must be jpeg or png format"
            )
    except AttributeError:
        raise HTTPException(status_code=400, detail="Image must be jpeg or png format")


def validate_registry(registry_dict):

    try:
        if not isinstance(registry_dict, dict):
            raise HTTPException(
                status_code=400, detail="Registry must be a dictionary"
            )
        key_elements = ["model", "loading_method", "class"]
        registry_dict_keys = list(registry_dict)

        for k in key_elements:
            if k not in registry_dict_keys:
                raise HTTPException(
                    status_code=400,
                    detail="Registry must contain model, loading_method, and class keys",
                )
        loading_method_keys = ["hub", "weights"]

        for k in registry_dict["loading_method"].keys():
            if k not in loading_method_keys:
                raise HTTPException(
                    status_code=400,
                    detail="loading_method must contain hub and weights keys"
                )

    except AttributeError:
        raise HTTPException(status_code=400, detail="Registry dictionary is not valid")
