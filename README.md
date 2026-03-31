# Single Handwritten Numerical Digit Classification  
A comparative study of CNN and ResNet18 for handwritten digit classification with detailed evaluation and analysis.


## Overview
- Task: Comparing CNN model built from scratch and a ResNet18 model in Handwritten Numerical Image Classification (from 0 to 9)
- Model: Convolutional Neural Network (CNN)
- Goal: Building a CNN model from scratch, comparing the model with ResNet18 and implementing an interactive demonstration  


## Why choosing CNN
CNN models are a strong baseline in image classification because of their ability to learn local spacial features effectively, and perform sufficiently even with a small dataset.  

In contrast, Vision Transformer (ViT) models, while outperform CNN models thanks to global dependencies, require significantly more data and computational resources to train effectively.  

Given the limited size of the dataset in this project, CNN is a more suitable and practical choice.  


## Demonstration
- A demonstration of the CNN model from scratch is produced at HuggingFace Space: https://huggingface.co/spaces/Fuyuki0312/ModelDetectingNumber-demo
- You may need to restart the space in order to use the model.
- Note: Input images are grayscale and their background color should be white by default.  
![description](Images/ModelDemonstration.jpg)  


## Metrics
- Model reached approximately 94% accuracy on a custom dataset.  

![description](Images/CNNAccuracyCurve.jpg) ![description](Images/CNNLostCurve.jpg)
![description](Images/CNNConfusionMatrix.jpg)  
(Confusion matrix collected model's prediction during validation after finishing training)
- The model sometimes confuses digits like 0, 3, 6, 8, and 9 because they share similar rounded shapes.  


## Dataset  

This project uses a custom dataset of handwritten digits (0–9).  
The original dataset: https://www.kaggle.com/datasets/olafkrastovski/handwritten-digits-0-9  

### Data Cleaning  

The dataset was manually inspected and cleaned to improve quality:  

- Removed corrupted images or ones that can be hardly seen
- Filtered out images where digits were not clearly visible

### Data Characteristics
- Image size: 90×140 (turned into grayscale afterward)
- Includes variations in handwriting styles and stroke thickness
- There is a number of digits which are not centered and have various size
- Some digits are visually similar (e.g., 0, 6, 8, 9), which introduces ambiguity

### Data Augmentation  
Heavy transformations (e.g., random rotation, large scaling) were avoided to preserve digit structure.  


## How to use this model
- If you wish to continue to train the existing model, consider to run the `train.py` with both `ModelDetectingNumber.pth` and `model.py` in the same directory. Hyperparameters in files can be changed to suit your need. Besides, if you wish to train a completely new model, simply delete or move file `ModelDetectingNumber.pth` away. When `ModelDetectingNumber.pth` is not found, `train.py` will automatically initialize a new model with architecture based on `model.py`.
- The dataset, used for training, should be put in the same directory with `train.py` under a folder named `numbers`, with the following structure:  
`numbers`/  
├── 0/  
│   ├── `img1.png`  
│   ├── `img2.png`  
│   └── ...  
├── 1/  
│   ├── `img1.png`  
│   └── ...  
├── 2/  
├── 3/  
├── 4/  
├── 5/  
├── 6/  
├── 7/  
├── 8/  
└── 9/  
- If you only want to use the model for inference, you can import model from `model.py` with weights loaded from `ModelDetectingNumber.pth`.
- Beisdes, `PlotConfusionMatrix.py` can be used to plot confusion matrix for current model with weights loaded from `ModelDetectingNumber.pth`.  

## Limitation
- Model usually gives right predictions only when the background color of input images is white because this model was trained primarily on numerical images with white backgrounds.
- If the input image is not so clear or handwritting is bad, the model may confidently produce a wrong prediction.


## Possible Improvements
- Expanding the dataset to include numerical images with diverse backgrounds (dark, textured, etc).
- Add more numerical images with bad handwritting to dataset.
