"""Machine Learning Model Training and Evaluation Script."""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


MODEL_NAME = "Random Forest Classifier"
DATASET_NAME = "Iris Dataset"


def load_data(test_size=0.2, random_state=42):
    """Load the Iris dataset and split into train and test sets."""
    iris = load_iris()
    X, y = iris.data, iris.target
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def train_model(X_train, y_train, random_state=42):
    """Train a Random Forest Classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Predict and calculate the model accuracy."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, predictions


def run_pipeline():
    """Execute complete end-to-end model pipeline."""
    print("=" * 50)
    print("           ML Pipeline Execution           ")
    print("=" * 50)
    print(f"Model Name   : {MODEL_NAME}")
    print(f"Dataset Used : {DATASET_NAME}")

    # 1. Load Data
    X_train, X_test, y_train, y_test = load_data()
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # 2. Train Model
    model = train_model(X_train, y_train)
    print("Model training completed successfully.")

    # 3. Evaluate and Make Predictions
    accuracy, predictions = evaluate_model(model, X_test, y_test)
    print(f"Sample predictions (first 5): {predictions[:5]}")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("=" * 50)

    return model, accuracy


if __name__ == "__main__":
    run_pipeline()
