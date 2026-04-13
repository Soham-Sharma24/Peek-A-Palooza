from urllib.request import urlopen
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as mp
import cv2 as cv
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load and preprocess CIFAR-10 dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# Class names for CIFAR-10 dataset
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Display the first 16 images from the training dataset
import sys
if '--visualize' in sys.argv:
    for i in range(16):
        mp.subplot(4, 4, i + 1)
        mp.xticks([])
        mp.yticks([])
        mp.imshow(training_images[i], cmap=mp.cm.binary)
        mp.xlabel(class_names[training_labels[i][0]])
    mp.show(block=True)

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Check if the model file exists
if os.path.exists('model.h5'):
    logging.info("Loading existing model...")
    model = tf.keras.models.load_model('model.h5')
else:
    logging.info("Training new model...")
    # Get number of epochs from environment variable or default to 10
    EPOCHS = int(os.getenv('EPOCHS', 10))
    # Compile and train the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(training_images, training_labels, epochs=EPOCHS, validation_data=(testing_images, testing_labels))
    # Save the trained model
    model.save('model.h5')

# Log model summary
model.summary()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Route to handle URI-based image processing
@app.route('/image-uri', methods=['POST'])
def image_uri():
    logging.info("Received request at /image-uri")
    try:
        data = request.get_json()
        uri = data.get('uri')
        
        # Log the URI to debug
        logging.info(f"URI Received: {uri}")

        if not uri:
            return jsonify({'error': 'URI not provided'}), 400

        # Attempt to open the image from the URI
        try:
            resp = urlopen(uri)
            logging.info(f"Response status code: {resp.status}")
        except Exception as e:
            logging.error(f"Error opening image: {str(e)}")
            return jsonify({'error': 'Failed to retrieve image from URI'}), 400

        # Read the image from the URI
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv.imdecode(image, cv.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Unable to decode image from URI'}), 400

        # Process and predict
        image = cv.resize(image, (32, 32))
        image = np.array(image).reshape(-1, 32, 32, 3) / 255.0
        prediction = model.predict(image)
        index = np.argmax(prediction)

        return jsonify({'result': class_names[index]})
    except Exception as e:
        logging.error(f"Error in /image-uri: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Route to handle direct image data prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'Image data not provided'}), 400

        img = np.array(data['image'])
        if img.shape[:2] != (32, 32):
            img = cv.resize(img, (32, 32))
        img = img.reshape(-1, 32, 32, 3) / 255.0

        # Make prediction
        prediction = model.predict(img)
        index = np.argmax(prediction)

        return jsonify({'result': class_names[index]})
    except Exception as e:
        logging.error(f"Error in /predict: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    # In production, use a WSGI server like Gunicorn:
    # gunicorn app:app --bind 0.0.0.0:5000 --workers 4
    app.run(debug=True)
