---
language: en
license: apache-2.0
tags:
- tomato
- leaf-disease
- agriculture
- image-classification
- scikit-learn
- random-forest
datasets:
- kaustubhb999/tomatoleaf
metrics:
- accuracy
---

# Tomato Leaf Disease Detection (Scikit-Learn Random Forest)

This model identifies 10 types of tomato leaf conditions (9 diseases + healthy).
It was developed as a robust alternative due to environment constraints with newer Python versions and PyTorch.

## Model Details
- **Type**: Random Forest Classifier
- **Features**: Flattened RGB images (64x64)
- **Classes**: 10
- **Accuracy**: ~65%

## Usage
The model is saved as a `joblib` file. You can load it using the `predict_sklearn.py` script provided in the repository.

```python
import joblib
model_data = joblib.load('tomato_leaf_sklearn_model.joblib')
# Use model_data['model'] to predict
```
