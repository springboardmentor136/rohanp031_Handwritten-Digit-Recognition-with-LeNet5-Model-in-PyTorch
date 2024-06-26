from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import numpy as np

# Define the MLP model architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Define the LeNet-5 model architecture
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 120)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize Flask app
app = Flask(__name__)

# Load the MLP model
input_size = 28 * 28
hidden_size = 500
num_classes = 10
mlp_model = MLP(input_size, hidden_size, num_classes)
mlp_model.load_state_dict(torch.load('mlp_model.pth', map_location=torch.device('cpu')))
mlp_model.eval()

# Load the CNN model
cnn_model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
cnn_model.load_weights('cnn_model.h5')

# Load the LeNet-5 model
lenet5_model = LeNet5()
lenet5_model.load_state_dict(torch.load('lenet5.pth', map_location=torch.device('cpu')))
lenet5_model.eval()

# Define transformation for incoming images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Prediction function for MLP
def predict_mlp(image_file):
    image_bytes = image_file.read()
    image = Image.open(BytesIO(image_bytes)).convert('L')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = mlp_model(image)
        _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

# Prediction function for CNN
def predict_cnn(image_file):
    try:
        image_bytes = image_file.read()
        image = Image.open(BytesIO(image_bytes)).convert('L')
        image = image.resize((28, 28))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        predictions = cnn_model.predict(image)
        return int(np.argmax(predictions[0]))
    except Exception as e:
        app.logger.error(f"Error in predict_cnn: {str(e)}")
        raise

# Prediction function for LeNet-5
def predict_lenet5(image_file):
    image_bytes = image_file.read()
    image = Image.open(BytesIO(image_bytes)).convert('L')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = lenet5_model(image)
        _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

# Define route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            file = request.files['file']
            model_type = request.form.get('model_type')

            if model_type == 'mlp':
                prediction = predict_mlp(file)
            elif model_type == 'cnn':
                prediction = predict_cnn(file)
            elif model_type == 'lenet5':
                prediction = predict_lenet5(file)
            else:
                return jsonify({'error': 'Invalid model type selected'}), 400

            return jsonify({'prediction': prediction})

    except Exception as e:
        app.logger.error(f"Error in prediction endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
