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
        # We first create the layers for our MLP model
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # We flatten the input image
        x = x.view(x.size(0), -1)
        # We apply the first fully connected layer followed by ReLU activation
        out = self.fc1(x)
        out = self.relu(out)
        # We then apply the second fully connected layer to get the output
        out = self.fc2(out)
        return out

# Initialize the Flask app
app = Flask(__name__)

# Load the model with predefined input size, hidden layer size, and number of classes
input_size = 28 * 28
hidden_size = 500
num_classes = 10
model = MLP(input_size, hidden_size, num_classes)
# Load the pre-trained model weights
model.load_state_dict(torch.load('mlp_model.pth', map_location=torch.device('cpu')))
# Set the model to evaluation mode
model.eval()

# Define the transformation for incoming images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure image is grayscale
    transforms.Resize((28, 28)),                  # Resize to 28x28 pixels
    transforms.ToTensor(),                        # Convert image to tensor
    transforms.Normalize((0.5,), (0.5,))          # Normalize with mean and std
])

# Define the prediction function
def predict_image(image_file):
    # We read the image file
    image_bytes = image_file.read()
    # We open and convert the image to grayscale
    image = Image.open(BytesIO(image_bytes)).convert('L')
    # We apply the transformations and add a batch dimension
    image = transform(image).unsqueeze(0)

    # Logging the intermediate steps for debugging
    print(f"Transformed Image Tensor: {image}")

    # We disable gradient calculation for inference
    with torch.no_grad():
        # We pass the image through the model to get outputs
        outputs = model(image)
        print(f"Model Outputs: {outputs}")

        # We get the predicted class
        _, predicted = torch.max(outputs.data, 1)
        print(f"Predicted Class: {predicted.item()}")
    return predicted.item()

# Define the route for the home page
@app.route('/')
def index():
    # We render the index.html template
    return render_template('index.html')

# Define the route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # We handle the POST request for image prediction
    if request.method == 'POST':
        # We get the uploaded file from the request
        file = request.files['file']
        # We use our prediction function to get the result
        prediction = predict_image(file)
        # We return the prediction result as JSON
        return jsonify({'prediction': prediction})

# We run the Flask app
if __name__ == '__main__':
    # We set debug mode to True for development
    app.run(debug=True)
