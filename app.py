from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image


import numpy as np
from PIL import Image
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Directory where uploaded files will be saved
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variable to store coordinates
coordinates = {}

@app.route('/')
def home_page():
    return "Testing -- home page"

# Route for uploading the image
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return jsonify({
            "message": "File successfully uploaded",
            "filepath": filepath
        }), 200
    
# Route to fetch and display an image
@app.route('/image/<filename>', methods=['GET'])
def get_image(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 404


# Route for receiving x and y coordinates
@app.route('/coordinates', methods=['POST'])
def upload_coordinates():
    x_coord = request.json.get('x_coordinate')
    y_coord = request.json.get('y_coordinate')

    if x_coord is None or y_coord is None:
        return jsonify({"error": "Missing x_coordinate or y_coordinate"}), 400

    coordinates['x_coordinate'] = x_coord
    coordinates['y_coordinate'] = y_coord

    return jsonify({
        "message": "Coordinates received successfully",
        "x_coordinate": x_coord,
        "y_coordinate": y_coord
    }), 200




#--ImageNET--
# Function to load and preprocess the image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


# Function to classify the image
def classify_image(img_path):
    img_array = load_and_preprocess_image(img_path)
    model = MobileNet(weights='imagenet')
    preds = model.predict(img_array)
    decoded_preds = decode_predictions(preds, top=3)[0]

    # Return only the description of the first prediction
    return str(decoded_preds[0][1])


# Route to classify the uploaded image
@app.route('/classify', methods=['POST'])
def upload_and_classify_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            predictions = classify_image(filepath)
            return jsonify({
                "message": "File successfully uploaded and classified",
                "predictions": predictions
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
