# Dog Breed Classification

## Overview
This repository is part of the Global AI Hub Aygaz Deep Learning Bootcamp and focuses on building a dog breed classification model using PyTorch. The code downloads and preprocesses the data from the Kaggle Dog Breed Identification competition and then trains two models: a custom VGG-16 model and a transfer learning-based ResNet model. The training process, including data loading, model creation, and training loops, is outlined in the notebook. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/egecandrsn/dog_breed_classification/blob/main/dog_breed_classification.ipynb)

## Data Preparation
The data is downloaded from Kaggle using provided URLs, including training images and labels. The images are organized into subdirectories based on their breed names, following the format required by torchvision's ImageFolder. The notebook visualizes random samples from the dataset to check for correctness and notes the class imbalance, prompting consideration for modified loss functions.

## Loading the Data
The data is transformed for model input using torchvision's ImageFolder and split into training and validation sets. DataLoader objects are created to efficiently load the data in batches during training.

## Model Creation
Two models are created: a custom VGG-16 model and a ResNet model with transfer learning. Class weights are computed to handle the imbalanced dataset.

## Training Models
The notebook provides a training function for PyTorch models, conducting training and validation loops, computing losses, and accuracies over specified epochs. The models are trained, and their performance is visualized using training and validation loss and accuracy plots.

## Making Predictions
A predict function is defined to use the trained models for making predictions. Random samples from the validation set are visualized along with their ground truth and predicted labels. Additionally, the models are used to make predictions from a sample image URL.

## Deployment
The models are saved as TorchScript for deployment in other environments. The label mapping is saved as a JSON file. A simple Streamlit app (`app.py`) is provided for interactive prediction using the trained model.

## Try it on Hugging Face Spaces
[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces-blueviolet)](https://huggingface.co/spaces/egecandrsn/dog_breed_classification)

## Requirements
The `requirements.txt` file lists the necessary libraries for running the code and the Streamlit app.

## Acknowledgments
This code is developed as part of the Global AI Hub Akbank Deep Learning Bootcamp. The dataset is sourced from the Kaggle Dog Breed Identification competition.

Feel free to explore the notebook and try out the Streamlit app!
