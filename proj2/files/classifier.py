"""
Project 2 – Data Classification Using AI
DecodeLabs Industrial Training Kit (2026)

Goal: Build a basic classification model using a small dataset (Iris).
Pipeline (IPO Framework):
  INPUT   -> Load Iris dataset, scale features
  PROCESS -> Train/test split, KNN algorithm
  OUTPUT  -> Confusion matrix, F1 score, accuracy
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)


def load_and_explore():
    """INPUT step 1: Load and understand the dataset."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df["species"] = df["target"].apply(lambda i: iris.target_names[i])

    print("=" * 60)
    print("STEP 1: LOAD & UNDERSTAND THE DATASET")
    print("=" * 60)
    print(f"Samples: {df.shape[0]} | Features: {iris.data.shape[1]} | Classes: {len(iris.target_names)}")
    print("\nClass distribution:")
    print(df["species"].value_counts())
    print("\nFirst 5 rows:\n", df.head())

    return iris.data, iris.target, iris.target_names


def scale_features(X):
    """INPUT step 2: Feature scaling (the 'Gatekeeper Rule')."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("\n" + "=" * 60)
    print("STEP 2: FEATURE SCALING (StandardScaler)")
    print("=" * 60)
    print("Mean after scaling (~0):", np.round(X_scaled.mean(axis=0), 3))
    print("Std after scaling (~1):", np.round(X_scaled.std(axis=0), 3))
    return X_scaled, scaler


def split_data(X, y):
    """PROCESS step 1: Train-test split (shuffle to remove order bias)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )
    print("\n" + "=" * 60)
    print("STEP 3: TRAIN-TEST SPLIT (80/20)")
    print("=" * 60)
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def find_best_k(X_train, y_train, X_test, y_test, max_k=20):
    """Tune K by scanning error rate — mirrors the 'elbow' slide."""
    errors = []
    for k in range(1, max_k + 1):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        errors.append(np.mean(preds != y_test))

    best_k = int(np.argmin(errors)) + 1
    print("\n" + "=" * 60)
    print("STEP 4: TUNING K (elbow scan)")
    print("=" * 60)
    for k, err in enumerate(errors, start=1):
        marker = "  <-- lowest error" if (k - 1) == np.argmin(errors) else ""
        print(f"K={k:2d}  error_rate={err:.3f}{marker}")
    print(f"\nBest K selected: {best_k}")
    return best_k


def train_and_predict(X_train, y_train, X_test, k):
    """PROCESS step 2: Instantiate, fit, predict (scikit-learn workflow)."""
    model = KNeighborsClassifier(n_neighbors=k)   # INSTANTIATE
    model.fit(X_train, y_train)                   # FIT
    predictions = model.predict(X_test)            # PREDICT
    return model, predictions


def evaluate(y_test, predictions, target_names):
    """OUTPUT: Confusion matrix, F1 score, accuracy — beyond raw accuracy."""
    print("\n" + "=" * 60)
    print("STEP 5: OUTPUT VALIDATION")
    print("=" * 60)

    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")

    print(f"Accuracy: {acc:.3f}")
    print(f"F1 Score (weighted): {f1:.3f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=target_names))


def main():
    X, y, target_names = load_and_explore()
    X_scaled, _ = scale_features(X)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    best_k = find_best_k(X_train, y_train, X_test, y_test)
    model, predictions = train_and_predict(X_train, y_train, X_test, best_k)
    evaluate(y_test, predictions, target_names)


if __name__ == "__main__":
    main()
