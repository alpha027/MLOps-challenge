from torchvision import transforms


MODEL_REGISTRY = {
    "densenet121": {
        "model": "DenseNet121",
        "loading_method": {
            "hub": {
                "url": "pytorch/vision:v0.10.0",
                "method": "load",
                "model": "densenet121",
                "weights": "IMAGENET1K_V1"
            },
            "weights": {
                "url": "https://download.pytorch.org/models/densenet121-a639ec97.pth",
                "path": "models/densenet121-a639ec97.pth",
                "str": "IMAGENET1K_V1"
            },
        },
        "class": "custom_imagenet.json",
        "transform": transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        ),
                    ]), 
    },
}