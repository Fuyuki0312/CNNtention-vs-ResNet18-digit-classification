# Single Handwritten Numerical Digit Classification
Image classification using Convolutional Neural Network, demonstrating inference pipeline with PyTorch.


## Overview
- Task: Numerical Image Classification (from 0 to 9)
- Model: Convolutional Neural Network (CNN)
- Challenges: Building a CNN model from scratch and implementing an interactive demonstration


## Demonstration
- A demonstration of the model is produced at HuggingFace Space: https://huggingface.co/spaces/Fuyuki0312/ModelDetectingNumber-demo
- You may need to restart the space in order to use the model.
- Note: Input images are grayscale and their background color should be white by default.


## Limitation
- Model usually gives right predictions only when the background color of input images is white because this model was trained primarily on numerical images with white backgrounds.
- If the input image is not so clear, the model may confidently produce a wrong prediction.


## Possible Improvements
- Expanding the dataset to include numerical images with diverse backgrounds (dark, textured, etc).
- Applying background-related data augmentation techniques during training.
