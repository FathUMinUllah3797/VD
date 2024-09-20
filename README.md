![image](https://github.com/user-attachments/assets/7d641685-3a9b-4f57-a1a3-13c8b2138ed1)# Dual Deep Learning Network for Abnormal Action Detection

This repository contains the implementation of the Dual Deep Learning Network for Abnormal Action Detection as described in the paper titled "[Dual Deep Learning Network for Abnormal Action Detection]([url](https://ieeexplore.ieee.org/document/10672568/authors#authors))."

## Overview

The code is structured to extract features from video frames using two deep learning architectures: LightFlowNet for optical flow features and a CNN for spatial features. The extracted features are then used to train a model for detecting abnormal actions in video sequences.





## File Descriptions

### FeatureExtraction.py


This script is responsible for extracting features from video frames using the LightFlowNet architecture for optical flow features and a CNN for spatial features. The extracted features are saved in the /Features directory.

### main.py

This script trains the model using the extracted features. It handles data loading, model architecture definition, training, and saving results such as accuracy and loss plots, classification reports, and confusion matrices in the /Results directory.

## Requirements

To run the code, you need the following Python packages:

numpy

scipy

tensorflow

matplotlib

seaborn

scikit-learn


You can install the required packages by running

pip install -r requirements.txt

## Acknowledgments
This work is based on the research presented in the paper "[Dual Deep Learning Network for Abnormal Action Detection]([url](https://ieeexplore.ieee.org/document/10672568/authors#authors))."
