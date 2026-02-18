import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import sys
import os

def predict(image_path, model_path="tomato_leaf_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint['classes']
    
    # Initialize model
    model = timm.create_model("mobilenetv3_large_100", num_classes=len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
    
    print(f"Prediction: {classes[predicted.item()]}")
    print(f"Confidence: {confidence.item()*100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        print("Usage: python predict.py <image_path>")
