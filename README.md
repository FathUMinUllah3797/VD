# Dual Deep Learning Network for Abnormal Action Detection

This repository contains the implementation of the Dual Deep Learning Network for Abnormal Action Detection as described in the paper titled [Dual Deep Learning Network for Abnormal Action Detection](https://ieeexplore.ieee.org/document/10672568/authors#authors)

## Overview

The code is structured to extract features from video frames using two deep learning architectures: LightFlowNet for optical flow features and a CNN for spatial features. The extracted features are then used to train the model for detecting abnormal actions in video sequences.

![image](https://github.com/user-attachments/assets/7d641685-3a9b-4f57-a1a3-13c8b2138ed1)

## Datasets

The following datasets are used in this article:

[RWF-2000](https://ieeexplore.ieee.org/abstract/document/9412502): A dataset containing various video sequences for action detection.

[Surveillance Fight](https://ieeexplore.ieee.org/abstract/document/8936070): A dataset specifically focused on fight scenarios in surveillance footage.

[Hockey Fight](https://link.springer.com/chapter/10.1007/978-3-642-23678-5_39): A dataset that includes hockey fight videos for action recognition.



## File Descriptions

### FeatureExtraction.py


This script is responsible for extracting spatial features from video frames. The extracted features are saved in the /Features directory.

#### Note 

For optical flow feature extraction, the script utilizes the LightFlowNet architecture. For more information, visit the [LightFlowNet GitHub repository](https://github.com/twhui/LiteFlowNet).

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

## Citation and Acknowledgements
<pre>
<code>
@article{,
  title={Dual Deep Learning Network for Abnormal Action Detection},
  author={Fath U Min Ullah, Zulfiqar Ahmad Khan, Sung Wook Baik, Estefania Talavera, Saeed Anwar, Khan Muhammad,},
  journal={  IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS)},
  year={2024},
  publisher={IEEE}
}</code>
</pre>

