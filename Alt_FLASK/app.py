from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model  # type: ignore
import numpy as np
from PIL import Image, ImageOps
import io
import base64

app = Flask(__name__)

# Load the saved models
mlp_model = load_model('mlp_model.h5')
lenet_model = load_model('lenet_model.h5')
cnn_model = load_model('cnn_model.h5')

# Define function to preprocess image
def preprocess_image(img):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to MNIST standard size
    img_array = np.array(img)  # Convert image to numpy array
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model input
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        file = request.files['file']
        img = Image.open(io.BytesIO(file.read()))
    else:
        img_data = request.form['image']
        img = Image.open(io.BytesIO(base64.b64decode(img_data.split(',')[1])))

    # Preprocess the image
    processed_img = preprocess_image(img)
    
    # Debug: Save the processed image to verify
    img.save("uploaded_image.png")
    Image.fromarray((processed_img[0].reshape(28, 28) * 255).astype(np.uint8)).save("processed_image.png")

    # Predict using models and get probabilities
    mlp_probs = mlp_model.predict(processed_img)[0]
    lenet_probs = lenet_model.predict(processed_img)[0]
    cnn_probs = cnn_model.predict(processed_img)[0]

    # Get predicted classes
    mlp_prediction = np.argmax(mlp_probs)
    lenet_prediction = np.argmax(lenet_probs)
    cnn_prediction = np.argmax(cnn_probs)

    response = {
        'mlp_prediction': int(mlp_prediction),
        'mlp_probability': float(mlp_probs[mlp_prediction]),
        'lenet_prediction': int(lenet_prediction),
        'lenet_probability': float(lenet_probs[lenet_prediction]),
        'cnn_prediction': int(cnn_prediction),
        'cnn_probability': float(cnn_probs[cnn_prediction])
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
