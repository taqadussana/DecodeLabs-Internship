# Project 2 — Data Classification Using AI

Second milestone of the DecodeLabs AI Industrial Training Kit (Batch 2026).
Builds a supervised-learning classifier on the **Iris dataset** using
**K-Nearest Neighbors (KNN)**.

## Goal
Build a basic classification model using a small dataset — load data,
split it, apply a classification algorithm, and validate the output.

## Pipeline (IPO Framework)
| Stage | What happens |
|---|---|
| **Input** | Load the Iris dataset (150 samples, 3 classes, 4 features) and scale it with `StandardScaler` |
| **Process** | Shuffle + split into train/test (80/20), scan K values to find the best one, train `KNeighborsClassifier`, predict |
| **Output** | Accuracy, weighted F1 score, confusion matrix, full classification report |

## Run it
```bash
pip install -r requirements.txt
python classifier.py
```

## Key files
- `classifier.py` — full pipeline, one function per stage
- `requirements.txt` — dependencies

## Notes
- Features are scaled before KNN because it's a distance-based algorithm —
  unscaled features (e.g. petal length in cm vs. a 0–1000 feature) would
  bias the distance calculation toward whichever feature has the larger
  numeric range.
- K is chosen by scanning error rate across K=1–20 and picking the elbow
  (lowest error, most stable region) rather than hardcoding it.
- Accuracy alone can be misleading on imbalanced data, so the confusion
  matrix and F1 score are reported too (Iris is balanced here, but the
  habit matters for future imbalanced datasets).
