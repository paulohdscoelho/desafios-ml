import os
import sys
from flask import Flask, request, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms

sys.path.append(os.path.abspath(os.path.join('../desafio-01/')))
from model import ResNet_pt

app = Flask(__name__)
PATH = "../desafio-01/output/k_cross_ResNet_state_dict.pt"

@app.route('/',methods=['GET','POST'])
def index():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = ResNet_pt().to(device)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    
    if request.method == 'POST':
        # Get the uploaded image file
        image_file = request.files['image']
        
        # Save the image to a temporary location
        image_path = "static/temp.jpg"
        image_file.save(image_path)

        # Preprocess the image
        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.46], std=[0.25])
            ])
        
        image = transform(image).unsqueeze(0).to(device)

        # Perform prediction using the model
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            predicted_label = "spoof" if predicted.item() == 0 else "live"                        
    
        return render_template('result.html', predicted_label=predicted_label, image_path=image_path)    
    
    return render_template('index.html')
