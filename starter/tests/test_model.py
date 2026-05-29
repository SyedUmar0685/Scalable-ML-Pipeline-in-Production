import os
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import (
    train_model, compute_model_metrics, inference, performance_on_categorical_slice
)


@pytest.fixture(scope="module")
def data():
    """Load cleaned census data."""
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "census_clean.csv"
    )
    return pd.read_csv(data_path)


@pytest.fixture(scope="module")
def trained_model(data):
    """Train model and return model, encoder, lb, and test data."""
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary",
        training=False, encoder=encoder, lb=lb
    )
    model = train_model(X_train, y_train)
    return model, X_test, y_test, encoder, lb


def test_train_model(trained_model):
    """Test that train_model returns a fitted model."""
    model, _, _, _, _ = trained_model
    assert model is not None
    assert hasattr(model, "predict")


def test_inference(trained_model):
    """Test that inference returns predictions with correct shape."""
    model, X_test, y_test, _, _ = trained_model
    preds = inference(model, X_test)
    assert preds.shape == y_test.shape
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics(trained_model):
    """Test that compute_model_metrics returns valid metrics."""
    model, X_test, y_test, _, _ = trained_model
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0


def test_performance_on_categorical_slice(data, trained_model):
    """Test performance_on_categorical_slice returns valid metrics."""
    model, _, _, encoder, lb = trained_model
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]
    p, r, fb = performance_on_categorical_slice(
        data, "education", "Bachelors", cat_features, "salary", encoder, lb, model
    )
    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0
    assert 0.0 <= fb <= 1.0


def test_performance_on_categorical_slice_empty(data, trained_model):
    """Test that slice with no data returns None."""
    model, _, _, encoder, lb = trained_model
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]
    p, r, fb = performance_on_categorical_slice(
        data, "education", "NONEXISTENT_VALUE", cat_features, "salary",
        encoder, lb, model
    )
    assert p is None
    assert r is None
    assert fb is None


def test_model_precision_above_threshold(trained_model):
    """Test that model precision is above a reasonable threshold."""
    model, X_test, y_test, _, _ = trained_model
    preds = inference(model, X_test)
    precision, _, _ = compute_model_metrics(y_test, preds)
    assert precision > 0.5
