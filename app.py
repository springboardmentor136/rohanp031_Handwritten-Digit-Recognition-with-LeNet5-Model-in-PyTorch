from flask import Flask, request, render_template
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from models.models import load_models

app = Flask(__name__)
cnn_model, lenet5_model, mlp_model = load_models()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def majority_voting(cnn_pred, lenet5_pred, mlp_pred):
    votes = np.bincount([cnn_pred, lenet5_pred, mlp_pred])
    return np.argmax(votes)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(file)
            img = transform(img).unsqueeze(0)
            
            cnn_output = cnn_model(img)
            lenet5_output = lenet5_model(img)
            mlp_output = mlp_model(img)
            
            _, cnn_pred = torch.max(cnn_output, 1)
            _, lenet5_pred = torch.max(lenet5_output, 1)
            _, mlp_pred = torch.max(mlp_output, 1)
            
            prediction = majority_voting(cnn_pred.item(), lenet5_pred.item(), mlp_pred.item())
            
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
