# Script to train machine learning model.
import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import (
    train_model, compute_model_metrics, inference, performance_on_categorical_slice
)

# Load in the cleaned data.
data = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "census_clean.csv"))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

# Evaluate model on test set.
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F-beta: {fbeta:.4f}")

# Save model artifacts.
model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
os.makedirs(model_dir, exist_ok=True)

with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(model_dir, "encoder.pkl"), "wb") as f:
    pickle.dump(encoder, f)

with open(os.path.join(model_dir, "lb.pkl"), "wb") as f:
    pickle.dump(lb, f)

print("Model saved to model/ directory.")

# Compute performance on slices and write to slice_output.txt
slice_output_path = os.path.join(os.path.dirname(__file__), "..", "slice_output.txt")
with open(slice_output_path, "w") as f:
    for feature in cat_features:
        for value in data[feature].unique():
            p, r, fb = performance_on_categorical_slice(
                test, feature, value, cat_features, "salary", encoder, lb, model
            )
            if p is not None:
                line = f"{feature} = {value}: Precision: {p:.4f} | Recall: {r:.4f} | F-beta: {fb:.4f}"
                print(line)
                f.write(line + "\n")

print(f"Slice output saved to {slice_output_path}")
