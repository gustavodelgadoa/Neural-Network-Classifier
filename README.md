# CIFAR-10 Neural Network Classifier

A Multilayer Perceptron (fully-connected neural network) classifier built with PyTorch to classify images from the CIFAR-10 dataset. This project implements a deep learning model using only linear (fully-connected) layers, without convolutional operations.

## Project Overview

This project implements a command-line neural network classifier that can:
- Train a multi-layer neural network on the CIFAR-10 dataset
- Achieve ≥45% test accuracy using only fully-connected layers
- Save trained models for later use
- Classify individual images from command-line input

## Determining Activation Functions for Hidden & output layers
Since we are only using linear layers and, are not using convolutional or recurrent layers. 

Activation Function - Hidden layers: 
 - Multilayer Perceptron : Rectified Linear Neurons (ReLu): 
    - Simple, fast computation
    - Prevents vanishing gradient
    - Industry standard for MLPs

Activation Function - Output layers: 
 - Multiclass Classification : Softmax Activation: 
    - CIFAR10 dataset contains 10 classes
    - each image belongs to exactly one class
## Preliminary

Download packages needed: pip install torch torchvision
