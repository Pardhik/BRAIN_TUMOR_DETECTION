from flask import Flask, jsonify, request
import keras
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout
from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.model_selection import train_test_split
import cv2
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import os
from flask_cors import CORS
import zipfile
from keras.models import load_model
from werkzeug.utils import secure_filename
app = Flask(__name__)
CORS(app)
model_path = "D:\\BTP\\backend\\trained_model_1709739464.h5"
model = load_model(model_path)
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
def preprocess_image(file):
    try:
        # Read image file from request
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise Exception("Image could not be loaded")
        # Resize the image
        img = cv2.resize(img, (150, 150))
        # Convert image to array and reshape
        img_array = np.array(img)
        img_array = img_array.reshape(1, 150, 150, 3)
        print("Image preprocessed successfully")
        return img_array
    except Exception as e:
        print("Error preprocessing image:", e)
        return None
@app.route('/extract_zip', methods=['GET'])
def extract_zip():
    # Get the absolute path to the directory where app.py is located
    base_dir = os.path.abspath(os.path.dirname(__file__))
# Path to the uploaded zip file
    zip_file_path = os.path.join(base_dir, "archive (24).zip")  # Use the correct filename
# Destination directory for extraction
    extract_dir = os.path.join(base_dir, "data")
# Extract the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    return "Extraction completed successfully"
@app.route('/list_files', methods=['GET'])
def list_files():
    # Get the absolute path to the directory where app.py is located
    base_dir = os.path.abspath(os.path.dirname(__file__))
 # Define the root directory for listing files
    root_dir = os.path.join(base_dir, "data")  # Use the extraction directory
# List to store file paths
    file_paths = []
    # Walk through the directory and list files
    for dirname, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_paths.append(os.path.join(dirname, filename))
    # Return the list of file paths as JSON response
    return jsonify(file_paths)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if request contains file
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        img_array = preprocess_image(file)
        
        if img_array is None:
            return jsonify({'error': 'Image preprocessing failed'})
        
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_label = labels[predicted_index]
        
        return jsonify({'predicted_index': int(predicted_index), 'predicted_label': predicted_label})
    except Exception as e:
        print("Error predicting:", e)
        return jsonify({'error': 'Prediction failed'})

if __name__ == "__main__":
    app.run(debug=True)
    
    
