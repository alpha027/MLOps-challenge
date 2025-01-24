import pytest

from app.services.validate import validate_registry
from ml.models.registry import MODEL_REGISTRY



def test_registry_integrity():

    registry_dict = MODEL_REGISTRY

    if not isinstance(registry_dict, dict):
        raise TypeError("Registry must be a dictionary")

    for k, v in registry_dict.items():
        validate_registry(v)

