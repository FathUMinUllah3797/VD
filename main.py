import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Flatten, Concatenate, Add, BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from matplotlib import pyplot as plt
import tensorflow as tf
import itertools
import pandas as pd
import scipy.io
# Function to plot and display confusion matrix
def plot_confusion_matrix(cm, class_labels, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels)
    plt.yticks(tick_marks, class_labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.grid(False)
    plt.savefig('Results/Confusion_Matrix.png', bbox_inches='tight')

# Function to plot training/validation accuracy and loss
def plot_training_history(train_acc, val_acc, train_loss, val_loss):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    # Plot Accuracy
    axes[0].plot(train_acc, color='royalblue', linewidth=2, label='Training Accuracy')
    axes[0].plot(val_acc, color='darkorange', linewidth=2, label='Validation Accuracy')
    axes[0].set_xlabel('Epochs', fontsize=14)
    axes[0].set_ylabel('Accuracy', fontsize=14)
    axes[0].legend(fontsize=12, loc='lower right')
    axes[0].grid(True)

    # Plot Loss
    axes[1].plot(train_loss, color='royalblue', linewidth=2, label='Training Loss')
    axes[1].plot(val_loss, color='darkorange', linewidth=2, label='Validation Loss')
    axes[1].set_xlabel('Epochs', fontsize=14)
    axes[1].set_ylabel('Loss', fontsize=14)
    axes[1].legend(fontsize=12, loc='upper right')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('Results/Training_History.png', bbox_inches='tight')

# Set up hyperparameters
num_classes = 2
batch_size = 32
epochs = 50

# Load CNN features and labels
cnn_features = np.load('path_to_cnn_features.npy').reshape(-1, 10, 1000)
cnn_labels = scipy.io.loadmat('path_to_cnn_labels.mat')['variable_name_for_labels']

# Load optical flow features and labels
optical_flow_features = np.load('path_to_optical_flow_features.npy').reshape(-1, 5, 1024)
optical_flow_labels = scipy.io.loadmat('path_to_optical_flow_labels.mat')['variable_name_for_labels']

# Train-test split
cnn_train, cnn_test, cnn_train_labels, cnn_test_labels = train_test_split(cnn_features, cnn_labels, test_size=0.2, random_state=42)
optical_train, optical_test, optical_train_labels, optical_test_labels = train_test_split(optical_flow_features, optical_flow_labels, test_size=0.2, random_state=42)

# Clear session to avoid clutter
tf.keras.backend.clear_session()

# Define CNN and Optical Flow Model inputs
input_cnn = Input(shape=(10, 1000))
input_optical = Input(shape=(5, 1024))

# CNN Feature Stream (GRU-based)
cnn_x1 = GRU(100, return_sequences=True)(input_cnn)
cnn_x2 = GRU(100, return_sequences=True)(cnn_x1)
cnn_out = Add()([cnn_x1, cnn_x2])  # Residual connection
cnn_x3 = GRU(100, return_sequences=False, dropout=0.1)(cnn_out)
cnn_dense = Dense(32, activation='relu')(cnn_x3)
cnn_flatten = Flatten()(cnn_dense)

# Optical Flow Feature Stream (GRU-based)
optical_x1 = GRU(100, return_sequences=True)(input_optical)
optical_x2 = GRU(100, return_sequences=True)(optical_x1)
optical_out = Add()([optical_x1, optical_x2])  # Residual connection
optical_x3 = GRU(100, return_sequences=False, dropout=0.1)(optical_out)
optical_dense = Dense(32, activation='relu')(optical_x3)
optical_flatten = Flatten()(optical_dense)

# Combine CNN and Optical Flow streams
merged = Concatenate()([cnn_flatten, optical_flatten])
merged_dense_1 = Dense(16, activation='relu')(merged)
merged_dense_2 = Dense(32, activation='relu')(merged_dense_1)
batch_norm = BatchNormalization()(merged_dense_2)
output_layer = Dense(num_classes, activation='softmax')(batch_norm)

# Compile Model
model = Model(inputs=[input_cnn, input_optical], outputs=output_layer)
optimizer = Adam(learning_rate=0.0001)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=optimizer, metrics=['accuracy'])

# Train the model
history = model.fit([cnn_train, optical_train], cnn_train_labels, epochs=epochs, validation_split=0.1, batch_size=batch_size)

# Save classification report
predictions = model.predict([cnn_test, optical_test])
class_names = ['Fight', 'Normal']
classification_rep = classification_report(cnn_test_labels.argmax(axis=1), predictions.argmax(axis=1), target_names=class_names, output_dict=True)
pd.DataFrame(classification_rep).transpose().to_csv('Results/Classification_Report.csv', index=True)

# Plot Accuracy and Loss
plot_training_history(history.history['accuracy'], history.history['val_accuracy'], history.history['loss'], history.history['val_loss'])

# Confusion Matrix
conf_matrix = confusion_matrix(cnn_test_labels.argmax(axis=1), predictions.argmax(axis=1))
plot_confusion_matrix(conf_matrix, class_names)

# ROC Curve
fpr, tpr, _ = roc_curve(cnn_test_labels.argmax(axis=1), predictions.argmax(axis=1))
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure()
plt.plot([0, 1], [0, 1], 'k--', color='darkorange', linewidth=2)
plt.plot(fpr, tpr, label='ROC curve (area = {:.3f})'.format(roc_auc), color='royalblue', linewidth=2)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve', fontsize=14)
plt.legend(loc='best', fontsize=14)
plt.savefig('Results/ROC_Curve.png', bbox_inches='tight')
