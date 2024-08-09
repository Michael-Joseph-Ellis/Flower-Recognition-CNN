# 🌸 Flower Recognition using Convolutional Neural Networks (CNN)

This repository contains the code and resources for a deep learning project focused on classifying images of flowers into distinct categories using a Convolutional Neural Network (CNN). The model is built using TensorFlow and Keras, with image preprocessing and augmentation techniques applied to improve model performance.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup Instructions](#setup-instructions)
- [Training the Model](#training-the-model)
- [Evaluation and Results](#evaluation-and-results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## 📁 Project Overview
Flower recognition is a common problem in computer vision that involves classifying images of flowers into specific categories. This project aims to:

- Load and preprocess image data from a custom dataset.
- Build a CNN model optimized for flower image classification.
- Train the model using data augmentation techniques to enhance generalization.
- Evaluate the model on a test dataset and analyze its performance.

## 🌼 Dataset
The dataset used in this project includes images of the following flower categories:

- **Dandelion**
- **Daisy**
- **Sunflower**
- **Tulip**
- **Rose**

Each category is stored in a separate directory within the primary data folder. The images are resized to a uniform dimension of 150x150 pixels for consistency.

### Data Preprocessing
- **Resizing:** All images are resized to 150x150 pixels.
- **Normalization:** Pixel values are normalized to a range of 0 to 1.
- **Augmentation:** Techniques such as rotation, zoom, and horizontal flipping are applied to increase the dataset size and improve model robustness artificially.

## 🧠 Model Architecture
The CNN model is built using the following architecture:

- **Input Layer:** 150x150x3 (for RGB images)
- **Convolutional Layers:** Multiple layers with ReLU activation and MaxPooling.
- **Flatten Layer:** Converts the 2D matrix to a 1D vector.
- **Fully Connected Layers:** Dense layers with ReLU activation.
- **Output Layer:** Softmax activation function for multi-class classification.

### Optimizer and Loss Function
- **Optimizer:** Adam with a learning rate of 0.001.
- **Loss Function:** Categorical Crossentropy for multi-class classification.

## 📊 Evaluation and Results
After training, the model is evaluated on a separate test set to assess its performance. The evaluation process includes measuring accuracy, generating a confusion matrix, and displaying some sample predictions.

### Key Evaluation Metrics
- **Accuracy:** The model's accuracy on the test set reflects the percentage of correctly classified images out of the total images.
- **Confusion Matrix:** A confusion matrix is used to visualize the performance of the model by showing the actual versus predicted classifications. This helps in identifying patterns where the model may be confusing one class for another.
- **Precision, Recall, and F1-Score:** These metrics provide a more detailed analysis of model performance, especially in the context of class imbalances. Precision measures the accuracy of the positive predictions, recall measures the model's ability to find all relevant cases within a class, and F1-Score provides a balance between precision and recall.

### Results Summary
- **Model Accuracy:** The model achieved an accuracy of **80.76%** on the test dataset, indicating a strong performance in classifying the various flower types.
- **Confusion Matrix Insights:** The confusion matrix revealed that the model occasionally confused similar-looking flowers, such as daisies and sunflowers. However, it performed well in distinguishing more distinct categories like tulips and roses.
- **Sample Predictions:** Below are some examples of the model's predictions on test images:

  - **Image 1:** Predicted - Tulip, Actual - Tulip
  - **Image 2:** Predicted - Dandelion, Actual - Dandelion
  - ...
  - ...
  - **Image 7:** Predicted - Sunflower, Actual - Dandelion (Misclassified)

These results suggest that while the model performs well overall, there is room for improvement in distinguishing between certain similar flower types. Fine-tuning the model or exploring more advanced architectures may further enhance its accuracy.

