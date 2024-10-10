from flask import Flask, request, jsonify
import numpy as np
from PIL import Image  # Importing PIL for image processing
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Initialize the Flask app
app = Flask(__name__)

# Set a limit for the uploaded file size (e.g., 16 MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Load the model from the local file system (included in the GitHub repo)
model_path = os.path.join(os.getcwd(), 'models', 'filter_classification_model.h5')
model = load_model(model_path)

# Dictionary of filter labels
filters = {
    0: "Sepia",
    1: "Black and White (B&W)",
    2: "Vintage",
    3: "Gaussian Blur",
    4: "Saturation Boost",
    5: "Vivid",
    6: "Warm",
    7: "Cool",
    8: "Soft Focus",
    9: "Film Grain",
    10: "Cartoon",
    11: "Dramatic"
}

# Define a route for the API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file is in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Open the image file and preprocess it for the model
        img = Image.open(file.stream)  # Use the stream to open the image
        img = img.resize((150, 150))  # Resize to the input shape expected by the model
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension

        # Perform prediction
        prediction = model.predict(img_array)
        predicted_label_index = np.argmax(prediction, axis=1)[0]  # Get the index of the predicted label
        
        # Map the predicted label index to the filter name
        predicted_filter = filters.get(predicted_label_index, "Unknown")

        # Return the prediction as a JSON response
        return jsonify({'prediction': predicted_filter})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
