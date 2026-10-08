from pathlib import Path

import pandas as pd
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "risk_model.pkl"

DATASET_PATH = BASE_DIR / "dataset" / "risk_data.csv"


# ============================================================
# CHECK DATASET
# ============================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_PATH}\n\n"
        "Please make sure risk_data.csv is inside the dataset folder."
    )


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("==========================================")
print("COASTAL RESILIENCE AI MODEL TRAINING")
print("==========================================")

print("\nLoading dataset...")
print(f"Dataset: {DATASET_PATH}")

data = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully.")
print("Rows:", len(data))
print("Columns:", len(data.columns))

print("\nColumns:")
print(data.columns.tolist())


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "rainfall",
    "wind_speed",
    "tide_level",
    "elevation",
    "population"
]

TARGET = "risk"


# ============================================================
# CHECK COLUMNS
# ============================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(missing_columns)
        + "\n\nExpected columns:\n"
        + ", ".join(required_columns)
    )


# ============================================================
# SELECT DATA
# ============================================================

data = data[required_columns].copy()


# ============================================================
# CONVERT TO NUMERIC
# ============================================================

for column in FEATURES:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

data[TARGET] = pd.to_numeric(
    data[TARGET],
    errors="coerce"
)


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

before = len(data)

data = data.dropna()

after = len(data)

print("\nInvalid rows removed:", before - after)
print("Valid rows:", after)


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = data[FEATURES]

y = data[TARGET].astype(int)


# ============================================================
# CHECK RISK CLASSES
# ============================================================

print("\nRisk classes:")
print(sorted(y.unique().tolist()))

valid_classes = {0, 1, 2}

if not set(y.unique()).issubset(valid_classes):
    raise ValueError(
        "\nThe risk column must contain:\n"
        "0 = Low Risk\n"
        "1 = Medium Risk\n"
        "2 = High Risk"
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# CREATE XGBOOST MODEL
# ============================================================

print("\nCreating XGBoost model...")

model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=3,
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss"
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Training completed successfully.")


# ============================================================
# TEST MODEL
# ============================================================

print("\nTesting model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==========================================")
print("MODEL RESULTS")
print("==========================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=[
            "Low Risk",
            "Medium Risk",
            "High Risk"
        ],
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving trained model...")

joblib.dump(
    model,
    MODEL_PATH
)


# ============================================================
# VERIFY MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "\nModel was not saved successfully."
    )


file_size = MODEL_PATH.stat().st_size


print("\n==========================================")
print("MODEL SAVED SUCCESSFULLY")
print("==========================================")

print(f"\nModel file:")
print(MODEL_PATH)

print(f"\nModel file size: {file_size} bytes")


# ============================================================
# SAMPLE PREDICTION
# ============================================================

print("\n==========================================")
print("SAMPLE PREDICTION")
print("==========================================")

sample = pd.DataFrame(
    {
        "rainfall": [120],
        "wind_speed": [60],
        "tide_level": [2.5],
        "elevation": [10],
        "population": [5000]
    }
)

prediction = model.predict(sample)

risk_labels = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}

predicted_class = int(prediction[0])

print("\nSample input:")
print(sample.to_string(index=False))

print(
    "\nPredicted risk:",
    risk_labels.get(
        predicted_class,
        "Unknown"
    )
)

print("\n==========================================")
print("TRAINING COMPLETED")
print("==========================================")