import os
from flask import Flask, redirect, render_template, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

# Define your ResNet18 model architecture with classes
num_classes = 2
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load the trained weights
model.load_state_dict(torch.load("resnet18_apple_disease.pth", map_location=torch.device('cpu')))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

def predict(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = transform(img)
        with torch.no_grad():
            output = model(img.unsqueeze(0))
            predicted_class = torch.argmax(output, dim=1).item()
        return predicted_class
    except Exception as e:
        return str(e)

@app.route('/submit', methods=['POST'])
def submit():
    image = request.files['image']
    filename = image.filename
    file_path = os.path.join('static/uploads', filename)
    image.save(file_path)

    pred = predict(file_path)
    if isinstance(pred, str):
        return jsonify({'error': pred})

    # Simple mapping for result
    result = "Healthy" if pred == 0 else "Unhealthy"

    return render_template('submit.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

