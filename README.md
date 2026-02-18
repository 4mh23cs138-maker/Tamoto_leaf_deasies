# Tomato Leaf Disease Detection

This project uses Machine Learning to identify various diseases in tomato leaves.

## Dataset
The model is trained on the [Tomato Leaf Disease Dataset](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf) from Kaggle.

## Model
Due to environment compatibility issues with PyTorch on Python 3.14, a **Random Forest Classifier** was trained using `scikit-learn` as a robust fallback.
- **Input Size**: 64x64 pixels (flattened)
- **Algorithm**: Random Forest (100 estimators)
- **Accuracy**: ~65% (on a subset of images)

## How to use

### Requirements
```bash
pip install scikit-learn pillow joblib numpy
```

### Prediction
To predict the disease of a leaf image:
```bash
python predict_sklearn.py <path_to_image>
```

## Supported Classes
1. Bacterial Spot
2. Early Blight
3. Healthy
4. Late Blight
5. Leaf Mold
6. Septoria Leaf Spot
7. Spider Mites (Two-Spotted Spider Mite)
8. Target Spot
9. Tomato Mosaic Virus
10. Tomato Yellow Leaf Curl Virus
