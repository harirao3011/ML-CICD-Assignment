"""Automated unit and integration tests for model.py."""

import numpy as np
import pytest
from model import DATASET_NAME, MODEL_NAME, evaluate_model, load_data, train_model


def test_metadata():
    """Verify model and dataset metadata constants."""
    assert MODEL_NAME == "Random Forest Classifier"
    assert DATASET_NAME == "Iris Dataset"


def test_load_data():
    """Verify dataset loading and splitting shapes."""
    X_train, X_test, y_train, y_test = load_data(test_size=0.2, random_state=42)

    assert len(X_train) > 0, "Training set should not be empty"
    assert len(X_test) > 0, "Test set should not be empty"
    assert len(X_train) + len(X_test) == 150, "Iris dataset total samples should be 150"
    assert X_train.shape[1] == 4, "Iris dataset should have 4 features"
    assert len(np.unique(y_train)) == 3, "Iris dataset has 3 classes"


def test_train_model():
    """Verify that model training completes and produces a fitted estimator."""
    X_train, _, y_train, _ = load_data()
    model = train_model(X_train, y_train)

    assert hasattr(model, "predict"), "Trained model must have a predict method"
    assert hasattr(model, "classes_"), "Trained model must have classes_ attribute"
    assert len(model.classes_) == 3, "Model must be trained on 3 distinct classes"


def test_model_accuracy_threshold():
    """Verify that the model meets or exceeds accuracy baseline of 90%."""
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    accuracy, predictions = evaluate_model(model, X_test, y_test)

    assert isinstance(accuracy, float), "Accuracy must be a float value"
    assert accuracy >= 0.90, f"Model accuracy ({accuracy}) should be at least 90%"
    assert len(predictions) == len(X_test), "Predictions length must match test set"


def test_prediction_output():
    """Verify that model predictions are within valid target classes."""
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    _, predictions = evaluate_model(model, X_test, y_test)

    valid_classes = {0, 1, 2}
    assert set(predictions).issubset(valid_classes), "Predictions must be valid class labels"
