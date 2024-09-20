import numpy
import cv2
import pickle
import numpy as np
import scipy.io as sio
from tensorflow.keras.models import Model
from tensorflow.keras import utils
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.efficientnet import preprocess_input
import os 
from tensorflow.keras.applications import EfficientNetB7

# Set parameters for the model and dataset
width = 600
height = 600
epochs = 20
input_shape = (height, width, 3)

# Load the EfficientNetB7 model pre-trained on ImageNet without the top layer (to be customized for the new task)
eff = EfficientNetB7(weights='imagenet', include_top=False, input_shape=input_shape)
eff.summary()

# Add custom layers to the model
x = eff.layers[-1].output  # Get the output of the last layer of EfficientNetB7
x = Flatten()(x)           # Flatten the output
x = Dense(1000)(x)         # Add a Dense layer with 1000 units (custom layer)
model = Model(inputs=eff.input, outputs=x)  # Define the new model with custom output

# Set the directory path to the dataset
dataset_directory = "Dataset"
dataset_folder = os.listdir(dataset_directory)  # List all the classes in the dataset folder

DatabaseFeautres = []  # Store extracted features from videos
DatabaseLabel = []     # Store labels corresponding to each video
cc = 0
vdno = 0

# Loop over each class directory in the dataset
for dir_counter in range(0, len(dataset_folder)):
    cc += 1  # Track class count
    single_class_dir = dataset_directory + "/" + dataset_folder[dir_counter]
    all_videos_one_class = os.listdir(single_class_dir)  # Get all video files in the current class directory
    
    # Loop through each video in the class directory
    for single_video_name in all_videos_one_class:
        print(cc)
        vdno += 1
        print('video_No : ', vdno)
        video_path = single_class_dir + "/" + single_video_name  # Get the video file path
        capture = cv2.VideoCapture(video_path)  # Open the video file
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames in the video
        video_features = []  # Store features extracted from frames of this video
        frames_counter = -1
        print('=======================================================')
        
        # Loop through each frame of the video
        while frames_counter < total_frames - 1:
            frames_counter += 1
            ret, frame = capture.read()  # Read a frame from the video
            if ret:  # Check if the frame was successfully read
                frame = cv2.resize(frame, (600, 600))  # Resize the frame to match input dimensions
                img_data = utils.img_to_array(frame)  # Convert the frame to an array
                img_data = np.expand_dims(img_data, axis=0)  # Add batch dimension
                img_data = preprocess_input(img_data)  # Preprocess the image data for EfficientNet
                single_featurevector = model.predict(img_data)  # Get the feature vector from the model
                video_features.append(single_featurevector)  # Store the feature vector
                
                # Every 10 frames, save the collected features and reset
                if frames_counter % 10 == 9:
                    temp = np.asarray(video_features)
                    print(temp.shape)
                    DatabaseFeautres.append(temp)  # Append the features to the database
                    DatabaseLabel.append(dataset_folder[dir_counter])  # Append the class label
                    video_features = []  # Reset the feature list for the next batch of frames

# Now prepare the feature and label data for storage
TotalFeatures = []
OneHotArray = []

# Flatten the feature vectors and store them in TotalFeatures
for sample in DatabaseFeautres:
    TotalFeatures.append(sample.reshape([1, 10000]))

# Convert TotalFeatures to a numpy array and reshape it for consistency
TotalFeatures = np.asarray(TotalFeatures)
TotalFeatures = TotalFeatures.reshape([len(DatabaseFeautres), 10000])

# Create one-hot encoded labels for the videos based on their classes
OneHotArray = []
kk = 1
for i in range(len(DatabaseFeautres) - 1):
    OneHotArray.append(kk)
    # If the class label changes, increment the class index
    if DatabaseLabel[i] != DatabaseLabel[i + 1]:
        kk += 1

# Initialize an array to store the one-hot encoded labels
OneHot = np.zeros([len(DatabaseFeautres), 2], dtype='int')

# Fill in the one-hot array based on the class index from OneHotArray
for i in range(len(DatabaseFeautres) - 1):
    print(i)
    OneHot[i, OneHotArray[i] - 1] = 1

# Save the features and labels to disk
np.save('Features/CNNFeatures', TotalFeatures)  # Save features as .npy file
sio.savemat('Features/CNNLabels.mat', mdict={'Labels': OneHot})  # Save labels as a .mat file
