import pytest

from app.services.validate import validate_registry
from ml.models.registry import MODEL_REGISTRY



def test_registry_integrity():

    registry_dict = MODEL_REGISTRY

    for k, v in registry_dict.items():
        validate_registry(v)

