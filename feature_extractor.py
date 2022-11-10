import numpy as np          # manipulating array
import tensorflow as tf     # tensorflow framework
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image            # Utilities for image preprocessing.
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# Class of Feature Extraction (with ResNet50 backbone)
class FeatureExtractor:

    # Define CNN model
    def __init__(self):           
        base_model = ResNet50(weights = "imagenet", include_top = False)    # Using ResNet50 with ImageNet weight
        model_input = base_model.input
        model_output = base_model.output   
        x = tf.keras.layers.GlobalAveragePooling2D()(model_output)      # Getting deep feature at avg pool: 1x1x2048)            
        self.model = Model(inputs = model_input, outputs = x)


    # Define feature extraction function
    def extract(self, img):
        x = preprocess_input(img)           # preprocess_input
        feature = self.model.predict(x)     # Deep feature at avg pool layer: (batch_size (1) , 2048)
        return feature