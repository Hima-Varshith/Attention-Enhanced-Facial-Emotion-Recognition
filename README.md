# Boosting Facial Emotion Recognition with Attention-Enhanced Residual Networks

This project was completed as part of **CSCI 5922 – Neural Networks and Deep Learning (NNDL)** during **Fall 2025** at the **University of Colorado Boulder**, taught by **Prof. Danna Gurari**.

The goal of the project was to improve facial emotion recognition on noisy and imbalanced datasets by building a lightweight CNN architecture that could better focus on important facial features, reduce overfitting, and improve recognition of subtle emotions.

## Project Motivation

Facial emotion recognition has applications in areas such as:

- Autism therapy
- Audience reaction analysis
- Expressive video communication
- Psychiatric evaluation

A major challenge in facial emotion recognition is distinguishing between subtle expressions while working with noisy and class-imbalanced datasets such as FER-2013.

To address this, the project combines **Residual Masking** with the **Convolutional Block Attention Module (CBAM)** to strengthen feature learning and help the model focus on more relevant spatial and channel-level information.

## What We Tried to Achieve

The main objective was to progressively improve a CNN-based facial emotion recognition model by:

- Establishing a baseline CNN
- Adding Residual Masking to improve feature propagation
- Integrating CBAM attention for spatial and channel-wise feature refinement
- Applying data augmentation and hyperparameter tuning
- Reducing overfitting and improving generalization
- Improving performance across all seven emotion classes, especially under class imbalance

The final architecture was designed to remain relatively lightweight while improving recognition performance on real-world facial emotion datasets.

## Datasets

The model was trained and evaluated using two facial emotion datasets:

### FER-2013

- Approximately **35,000 grayscale facial images**
- Common benchmark for facial emotion recognition
- Contains noisy and challenging samples

### RAF-DB

- Approximately **15,000 real-world facial images**
- Provides greater variation in expressions and real-world conditions

Both datasets cover seven emotion classes:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

All images were standardized to **48 × 48** resolution.

### Data Split

The combined training data was split into:

- **36,882 training images**
- **4,098 validation images**

The official combined test sets were kept separate for final evaluation:

- **10,246 test images**

## Model Architecture

The final model consists of three convolutional blocks.

Each block follows:

```text
Conv2D
→ BatchNorm
→ ReLU
→ Residual Mask Block
→ CBAM
→ Max Pooling
