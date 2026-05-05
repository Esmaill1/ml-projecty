import os
import base64
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
from flask import Flask, request, jsonify, render_template
import io

app = Flask(__name__)

# Load the trained digit recognition model without compiling
model_path = os.path.join(os.path.dirname(__file__), 'digit_recognizer_improved110.h5')
model = load_model(model_path, compile=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # image data is a base64 string "data:image/png;base64,..."
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        
        # convert to numpy array
        img = Image.open(io.BytesIO(img_bytes))
        img = np.array(img)
        
        # Convert RGBA to grayscale
        img = Image.fromarray(img).convert('L')
        img = np.array(img)

        # Resize and normalize
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        img = 1 - img  # Invert colors: white bg, black digits
        img_input = img.reshape(1, 28, 28, 1)

        # Only predict if the user actually drew something
        if np.count_nonzero(img > 0.1) > 10:
            prediction = model.predict(img_input, verbose=0)
            digit = str(np.argmax(prediction))
            return jsonify({'prediction': digit, 'success': True})
        else:
            return jsonify({'error': 'Canvas too empty', 'success': False})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5006)
