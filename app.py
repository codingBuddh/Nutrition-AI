from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
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

# Global variable to store coordinates (in a production app, you'd likely use a database)
coordinates = {}

@app.route('/')
def home_page():
    return "Testing -- home page"

# Route for uploading the image
@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if 'file' is part of the request
    if 'file' not in request.files:
        print("File part missing")
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']

    # Check if the filename is provided
    if file.filename == '':
        print("No file selected")
        return jsonify({"error": "No file selected for uploading"}), 400

    # Save the file to the uploads folder
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        print(f"File uploaded and saved to {filepath}")
        
        return jsonify({
            "message": "File successfully uploaded",
            "filepath": filepath
        }), 200

# Route for receiving x and y coordinates
@app.route('/coordinates', methods=['POST'])
def upload_coordinates():
    x_coord = request.json.get('x_coordinate')
    y_coord = request.json.get('y_coordinate')

    # Validate x and y coordinates
    if x_coord is None or y_coord is None:
        print("Missing coordinates")
        return jsonify({"error": "Missing x_coordinate or y_coordinate"}), 400

    # Store coordinates
    coordinates['x_coordinate'] = x_coord
    coordinates['y_coordinate'] = y_coord

    print(f"Received coordinates: x = {x_coord}, y = {y_coord}")
    return jsonify({
        "message": "Coordinates received successfully",
        "x_coordinate": x_coord,
        "y_coordinate": y_coord
    }), 200

# --------------------------ImageNet (MobileNetV2)---------------------------
# Loading the MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Function to load and preprocess the image
def load_and_preprocess_image(img_path):
    print(f"Loading image from {img_path}")
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    print("Image successfully preprocessed")
    return img_array

# Function to classify the image
def classify_image(img_path):
    print(f"Classifying image at {img_path}")
    img_array = load_and_preprocess_image(img_path)

    # Predict
    preds = model.predict(img_array)
    # print(f"Prediction raw output: {preds}")

    # Decode the results into a list of tuples (class, description, probability)
    decoded_preds = decode_predictions(preds, top=3)[0]
    print(f'Predicted: {decoded_preds}')

    # Convert the predictions to standard Python types
    converted_preds = [
        {
            "class": str(pred[0]),  # class label (string)
            "description": str(pred[1]),  # description (string)
            "probability": float(pred[2])  # probability (convert float32 to float)
        }
        for pred in decoded_preds
    ]
    return converted_preds

# API route to upload an image and classify it
@app.route('/classify', methods=['POST'])
def upload_and_classify_image():
    # Check if 'file' is part of the request
    if 'file' not in request.files:
        print("File part missing")
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']

    # Check if the filename is provided
    if file.filename == '':
        print("No file selected")
        return jsonify({"error": "No file selected for uploading"}), 400

    # Save the file to the uploads folder
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        print(f"File uploaded and saved to {filepath}")

        # Classify the uploaded image
        try:
            predictions = classify_image(filepath)
            print(f"Classification completed for {filepath}")
            
            return jsonify({
                "message": "File successfully uploaded and classified",
                "predictions": predictions
            }), 200
        except Exception as e:
            print(f"Error during classification: {e}")
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)