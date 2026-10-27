# Step 1: Import required libraries

import os
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# Step 2: Define file paths

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset" / "house_prices.csv"

MODEL_PATH = BASE_DIR / "model.joblib"

FEATURE_PATH = BASE_DIR / "feature_names.joblib"


# Step 3: Load dataset

print("\nLoading House Price Dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. "
        "Place house_prices.csv inside the dataset folder."
    )

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(15))

print("\nDuplicate rows:")
print(df.duplicated().sum())


# Step 4: Select target column

TARGET_COLUMN = "SalePrice"

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' was not found.\n"
        f"Available columns: {df.columns.tolist()}\n"
        "Update TARGET_COLUMN in train.py to match your dataset."
    )


# Step 5: Remove rows where target is missing

df = df.dropna(subset=[TARGET_COLUMN]).copy()

# Convert target to numeric values
df[TARGET_COLUMN] = pd.to_numeric(
    df[TARGET_COLUMN],
    errors="coerce"
)

df = df.dropna(subset=[TARGET_COLUMN])

# Remove rows with invalid or non-positive house prices
df = df[df[TARGET_COLUMN] > 0].copy()

# Remove duplicate records
df = df.drop_duplicates().copy()


# Step 6: Separate features and target

# Remove the target column from the input features.
# Id is also removed because it is an identifier,
# not a meaningful house characteristic.

columns_to_drop = [TARGET_COLUMN]

if "Id" in df.columns:
    columns_to_drop.append("Id")

X = df.drop(columns=columns_to_drop)

y = df[TARGET_COLUMN]


# Remove completely empty feature columns

X = X.dropna(axis=1, how="all")

if X.shape[1] == 0:
    raise ValueError("No usable input features were found.")

feature_names = X.columns.tolist()

print("\nNumber of input features:", len(feature_names))
print("\nTarget column:", TARGET_COLUMN)


# Step 7: Identify numerical and categorical columns

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical features:", len(numerical_features))

print("Categorical features:", len(categorical_features))


# Step 8: Create preprocessing pipelines

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ],
    remainder="drop"
)


# Step 9: Split dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", X_train.shape[0])

print("Testing records:", X_test.shape[0])


# Step 10: Define regression algorithms

models = {
    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(alpha=1.0),

    "Decision Tree Regression": DecisionTreeRegressor(
        max_depth=12,
        random_state=42
    ),

    "Random Forest Regression": RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting Regression": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# Step 11: Train and evaluate models

results = []

best_model = None
best_model_name = None
best_r2 = -np.inf

print("\n" + "=" * 65)
print("REGRESSION MODEL TRAINING AND EVALUATION")
print("=" * 65)

for model_name, regressor in models.items():

    print(f"\nTraining: {model_name}")

    # Keep preprocessing and model together in one pipeline

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ]
    )

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predict on unseen test data
    y_pred = pipeline.predict(X_test)

    # Calculate evaluation metrics
    r2 = r2_score(y_test, y_pred)

    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R2 Score: {r2:.4f}")

    print(f"MAE: {mae:,.2f}")

    print(f"RMSE: {rmse:,.2f}")

    results.append({
        "Model": model_name,
        "R2 Score": r2,
        "MAE": mae,
        "RMSE": rmse
    })

    # Select the best model based on test R2

    if np.isfinite(r2) and r2 > best_r2:

        best_r2 = r2

        best_model = pipeline

        best_model_name = model_name


# Step 12: Display model comparison

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(results_df.to_string(index=False))


# Step 13: Save the best model

if best_model is None:
    raise RuntimeError("No model was successfully trained.")

joblib.dump(best_model, MODEL_PATH)

joblib.dump(feature_names, FEATURE_PATH)


# Step 14: Display final results

print("\n" + "=" * 65)

print("BEST MODEL:", best_model_name)

print(f"BEST TEST R2 SCORE: {best_r2:.4f}")

print("\nSaved model:", MODEL_PATH)

print("Saved feature names:", FEATURE_PATH)

print("\nTraining completed successfully!")