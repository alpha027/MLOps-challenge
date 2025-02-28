# MLOps challenge

This is a small minimal AI inference API for image classification.

## Development Requirements

- Python3.11.0
- Pip
- Poetry (Python Package Manager)

## Installation

```sh
python3.11 -m venv venv # 3.12 works as well
source venv/bin/activate
make install
```

## Runnning Localhost

`make run`

## Deploy app

`make deploy`

## Running Tests

`make test`

## Usecase

### POST `/predict`

A POST request can be made using the following command:
```
curl -X POST -H "Content-Type: application/json" -d '{"image": "Image Content goes Here"}' http://localhost:5000/predict
```
This returns a JSON response of the following form:
```
{"response": "class_name"}
```

### POST `/predict-form-data`

A POST request can be made such that the image is sent as multipart/form-data using the --form option.
The request can be done using the following command:
```
curl --location 'http://localhost:5000/predict-form-data' \
--form 'image=@"file_path_of_image.jpeg"'
```
This returns a JSON response of the following form:
```
{"response": "class_name"}
```


## Access Swagger Documentation

> <http://localhost:5000/docs>

## Access Redocs Documentation

> <http://localhost:5000/redoc>

### M.L Model Environment

```sh
MODEL_PATH=./ml/model/
MODEL_NAME=densenet121
```

## Project structure

Files related to application are in the `app` or `tests` directories.
Application parts are:

    app
    |
    | # Fast-API stuff
    ├── api                 - web related stuff.
    │   └── routes          - web routes.
    ├── core                - application configuration, startup events, logging.
    ├── models              - Data models for this application.
    ├── services            - logic that is not just crud related.
    ├── main-aws-lambda.py  - [Optional] FastAPI application for AWS Lambda creation and configuration.
    └── main.py             - FastAPI application creation and configuration.
    |
    │
    ├── notebooks        - Jupyter notebooks. Naming convention is a number (for ordering),
    |
    ├── ml               - AI related logic for AI models and classes
    │
    └── tests            - pytest

## AWS

Deploying inference service to AWS Lambda

### Authenticate

1. Install `awscli` and `sam-cli`
2. `aws configure`

### Deploy to Lambda

1. Run `sam build`
2. Run `sam deploy --guiChange this portion for other types of models

## Add the correct type hinting when completed

`aws cloudformation delete-stack --stack-name <STACK_NAME_ON_CREATION>`

Made by <https://github.com/arthurhenrique/cookiecutter-fastapi/graphs/contributors> with ❤️


## References

This pipeline was generated using the [Cookiecutter](https://github.com/arthurhenrique/cookiecutter-fastapi) python module.
The AI-based classification relies on the [PyTorch](https://pytorch.org/) framework.
