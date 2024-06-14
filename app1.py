import os
import pickle
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template
from models.models import MLP  # Import the MLP class

# Initialize Flask application
app = Flask(__name__)

# Path to the model
MODEL_PATH = 'models/mlp_model.pkl'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformation
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Check if the model file exists and load the model
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        loaded_model = pickle.load(f)
        loaded_model.to(device)
        loaded_model.eval()
else:
    raise FileNotFoundError(f'Model file not found: {MODEL_PATH}')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(file)
            img = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = loaded_model(img)
                _, predicted = torch.max(outputs.data, 1)
                prediction = predicted.item()

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
