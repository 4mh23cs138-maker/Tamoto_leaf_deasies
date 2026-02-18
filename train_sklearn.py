import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
import time

DATA_DIR = r"C:\leaf_data\tomato\train"
IMG_SIZE = (64, 64)
MAX_SAMPLES_PER_CLASS = 200 # For speed since we are in a terminal environment

def load_data():
    X = []
    y = []
    classes = sorted(os.listdir(DATA_DIR))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    print(f"Loading data from {DATA_DIR}...")
    for cls in classes:
        cls_path = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cls_path):
            continue
        
        count = 0
        img_names = os.listdir(cls_path)
        print(f"  Processing {cls} ({len(img_names)} images)...")
        
        for img_name in img_names:
            if count >= MAX_SAMPLES_PER_CLASS:
                break
            
            img_path = os.path.join(cls_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(IMG_SIZE)
                X.append(np.array(img).flatten())
                y.append(class_to_idx[cls])
                count += 1
            except Exception as e:
                print(f"    Failed to load {img_name}: {e}")
                
    return np.array(X), np.array(y), classes

def main():
    start_time = time.time()
    
    X, y, classes = load_data()
    print(f"Data Loaded: {X.shape}, labels: {y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Save model and metadata
    model_data = {
        'model': model,
        'classes': classes,
        'img_size': IMG_SIZE
    }
    joblib.dump(model_data, 'tomato_leaf_sklearn_model.joblib')
    print(f"Model saved to tomato_leaf_sklearn_model.joblib")
    
    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
