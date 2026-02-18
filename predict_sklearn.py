import joblib
import numpy as np
from PIL import Image
import sys
import os

def predict(image_path, model_path="tomato_leaf_sklearn_model.joblib"):
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    # Load model data
    model_data = joblib.load(model_path)
    model = model_data['model']
    classes = model_data['classes']
    img_size = model_data['img_size']
    
    # Load and preprocess image
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(img_size)
        img_array = np.array(img).flatten().reshape(1, -1)
        
        # Predict
        prediction = model.predict(img_array)[0]
        probabilities = model.predict_proba(img_array)[0]
        confidence = probabilities[prediction]
        
        print(f"Prediction: {classes[prediction]}")
        print(f"Confidence: {confidence*100:.2f}%")
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        print("Usage: python predict.py <image_path>")
