from flask import Flask, request, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms

from model import ResNet_pt

app = Flask(__name__)
PATH = "static/k_cross_ResNet_balanced_state_dict.pt"

@app.route('/',methods=['GET','POST'])
def index():
    
    device = torch.device("cpu")
    
    model = ResNet_pt().to(device)
    model.load_state_dict(torch.load(PATH,map_location=device))
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
            confidence = torch.nn.functional.softmax(output, dim=1)[0] * 100
            confidence = round(confidence[predicted.item()].item(),2)                                   
    
        return render_template('result.html', predicted_label=predicted_label, image_path=image_path, confidence=confidence)    
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
