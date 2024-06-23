from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO

# Define the model architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input images
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Initialize the Flask app
app = Flask(__name__)

# Load the model
input_size = 28 * 28
hidden_size = 500
num_classes = 10
model = MLP(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load('mlp_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define the transformation for incoming images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure image is grayscale
    transforms.Resize((28, 28)),                  # Resize to 28x28
    transforms.ToTensor(),                        # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))          # Normalize with mean and std
])

# Prediction function
def predict_image(image_file):
    image_bytes = image_file.read()
    image = Image.open(BytesIO(image_bytes)).convert('L')  # Convert to grayscale
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Logging intermediate steps
    print(f"Transformed Image Tensor: {image}")

    with torch.no_grad():
        outputs = model(image)
        print(f"Model Outputs: {outputs}")

        _, predicted = torch.max(outputs.data, 1)
        print(f"Predicted Class: {predicted.item()}")
    return predicted.item()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        prediction = predict_image(file)
        return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
