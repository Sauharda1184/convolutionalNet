# 🌌 Classifying Galaxies Using Convolutional Neural Networks

![Python](https://img.shields.io/badge/Python-3.8-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📖 Overview

Astronomers receive vast quantities of image data from telescopes every day. However, until these images are annotated, their scientific utility remains limited. This project tackles the challenge of automatic galaxy classification using deep learning techniques.

Using the **Galaxy Zoo** dataset — a crowd-sourced astronomical image collection — I built a **Convolutional Neural Network (CNN)** capable of classifying galaxy images into four morphological categories, including galaxies with unusual or "odd" features.

## 📁 Dataset

- **Source**: [Galaxy Zoo](https://www.zooniverse.org/projects/zookeeper/galaxy-zoo)
- **Input**: Grayscale or RGB images of galaxies
- **Classes** (One-hot Encoded):
  - `[1, 0, 0, 0]`: Galaxies with no identifying characteristics
  - `[0, 1, 0, 0]`: Spiral galaxies
  - `[0, 0, 1, 0]`: Elliptical galaxies
  - `[0, 0, 0, 1]`: Irregular or "odd" galaxies

## 🧠 Model Architecture

- Input Layer (normalized image data)
- Multiple **Conv2D + MaxPooling** layers
- **Dropout** and **Batch Normalization** for regularization
- Fully connected **Dense** layers with **ReLU** activation
- **Softmax** output layer for multiclass classification

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
