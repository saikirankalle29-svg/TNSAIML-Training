# House Price Prediction Using Regression Algorithms

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# --------------------------------------------------
# STEP 1: PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# --------------------------------------------------
# STEP 2: FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "dataset" / "house_prices.csv"

MODEL_PATH = BASE_DIR / "model.joblib"

FEATURE_PATH = BASE_DIR / "feature_names.joblib"


# --------------------------------------------------
# STEP 3: LOAD DATA AND MODEL
# --------------------------------------------------

@st.cache_data
def load_dataset():

    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_names():

    return joblib.load(FEATURE_PATH)


# --------------------------------------------------
# STEP 4: SIDEBAR
# --------------------------------------------------

st.sidebar.title("🏠 House Price Predictor")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Dataset Explorer",
        "Model Performance",
        "Predict House Price"
    ]
)


# --------------------------------------------------
# STEP 5: CHECK REQUIRED FILES
# --------------------------------------------------

if not DATA_PATH.exists():

    st.error(
        "Dataset not found. Place house_prices.csv "
        "inside the dataset folder."
    )

    st.stop()


if not MODEL_PATH.exists() or not FEATURE_PATH.exists():

    st.warning(
        "The trained model has not been created yet. "
        "Run the training script first."
    )

    st.code(
        "python dataset/train.py",
        language="powershell"
    )

    st.stop()


# Load required files

try:

    df = load_dataset()

    model = load_model()

    feature_names = load_feature_names()

except Exception as exc:

    st.error(f"Unable to load the project files: {exc}")

    st.stop()


TARGET_COLUMN = "SalePrice"


# Validate target column

if TARGET_COLUMN not in df.columns:

    st.error(
        f"Target column '{TARGET_COLUMN}' "
        "was not found in the dataset."
    )

    st.stop()


# Prepare data for dashboard statistics

df = df.dropna(subset=[TARGET_COLUMN]).copy()

df[TARGET_COLUMN] = pd.to_numeric(
    df[TARGET_COLUMN],
    errors="coerce"
)

df = df.dropna(subset=[TARGET_COLUMN])

df = df[df[TARGET_COLUMN] > 0].copy()


# --------------------------------------------------
# STEP 6: DASHBOARD PAGE
# --------------------------------------------------

if page == "Dashboard":

    st.title("🏠 House Price Prediction")

    st.subheader(
        "Using Regression Algorithms"
    )

    st.markdown(
        """
        This project uses machine learning regression
        algorithms to estimate house prices based on
        available property characteristics.

        The dashboard supports dataset exploration,
        model comparison, and interactive price prediction.
        """
    )

    st.markdown("---")

    # Summary metrics

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Houses",
        f"{len(df):,}"
    )

    col2.metric(
        "Input Features",
        len(feature_names)
    )

    col3.metric(
        "Average House Price",
        f"{df[TARGET_COLUMN].mean():,.0f}"
    )

    col4.metric(
        "Median House Price",
        f"{df[TARGET_COLUMN].median():,.0f}"
    )

    st.markdown("---")

    # Dataset preview

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    # House price distribution

    st.subheader("House Price Distribution")

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.hist(
        df[TARGET_COLUMN],
        bins=30,
        edgecolor="black"
    )

    ax.set_xlabel("House Price")

    ax.set_ylabel("Number of Houses")

    ax.set_title("Distribution of House Prices")

    st.pyplot(fig)

    plt.close(fig)

    st.info(
        "Use the sidebar to explore the dataset, "
        "compare regression models, or predict a house price."
    )


# --------------------------------------------------
# STEP 7: DATASET EXPLORER
# --------------------------------------------------

elif page == "Dataset Explorer":

    st.title("📊 Dataset Explorer")

    st.write(
        "Explore the dataset used to train the regression models."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rows",
        df.shape[0]
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    col3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    st.markdown("---")

    st.subheader("Dataset Records")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.subheader("Missing Values by Column")

    missing_values = (
        df.isnull()
        .sum()
        .sort_values(ascending=False)
    )

    missing_df = missing_values.reset_index()

    missing_df.columns = [
        "Column",
        "Missing Values"
    ]

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    st.subheader("Correlation with House Price")

    numeric_df = df.select_dtypes(
        include=["number"]
    )

    if TARGET_COLUMN in numeric_df.columns:

        correlations = (
            numeric_df.corr()[TARGET_COLUMN]
            .drop(TARGET_COLUMN)
            .dropna()
            .sort_values(ascending=False)
        )

        if not correlations.empty:

            st.bar_chart(correlations)

        else:

            st.info(
                "No valid numerical correlations are available."
            )


# --------------------------------------------------
# STEP 8: MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.title("📈 Regression Model Performance")

    st.write(
        "Compare the regression algorithms using "
        "the same training and testing split."
    )

    X = df[feature_names].copy()

    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

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

    results = []

    with st.spinner("Evaluating regression models..."):

        for name, regressor in models.items():

            from sklearn.base import clone

            preprocessor = clone(
                model.named_steps["preprocessor"]
            )

            from sklearn.pipeline import Pipeline

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("regressor", regressor)
                ]
            )

            pipeline.fit(
                X_train,
                y_train
            )

            predictions = pipeline.predict(
                X_test
            )

            results.append({

                "Model": name,

                "R2 Score": r2_score(
                    y_test,
                    predictions
                ),

                "MAE": mean_absolute_error(
                    y_test,
                    predictions
                ),

                "RMSE": np.sqrt(
                    mean_squared_error(
                        y_test,
                        predictions
                    )
                )
            })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="R2 Score",
        ascending=False
    )

    best_result = results_df.iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Best Algorithm",
        best_result["Model"]
    )

    col2.metric(
        "Best R² Score",
        f"{best_result['R2 Score']:.4f}"
    )

    col3.metric(
        "Best Model MAE",
        f"{best_result['MAE']:,.2f}"
    )

    st.subheader("Model Comparison Table")

    st.dataframe(
        results_df,
        use_container_width=True
    )

    st.subheader("R² Score Comparison")

    chart_df = results_df.set_index("Model")

    st.bar_chart(
        chart_df["R2 Score"]
    )

    st.subheader("Error Comparison")

    error_df = results_df.set_index("Model")[
        ["MAE", "RMSE"]
    ]

    st.bar_chart(error_df)

    st.caption(
        "R² measures the model's goodness of fit. "
        "MAE and RMSE measure prediction errors "
        "in the target price's units. "
        "These metrics are calculated on the held-out test set."
    )


# --------------------------------------------------
# STEP 9: HOUSE PRICE PREDICTION
# --------------------------------------------------

elif page == "Predict House Price":

    st.title("🏡 Predict House Price")

    st.markdown(
        """
        Enter the property details below to estimate
        its selling price using the trained regression model.
        """
    )

    st.info(
        "The prediction is an estimate based on the "
        "training dataset. It is not a guaranteed market price."
    )

    st.markdown("---")

    # Determine numerical and categorical feature names

    training_df = df[feature_names]

    numerical_features = training_df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = training_df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Split inputs into two columns

    left_col, right_col = st.columns(2)

    input_values = {}

    with st.form("house_prediction_form"):

        st.subheader("Property Information")

        left_col, right_col = st.columns(2)

        for index, feature in enumerate(feature_names):

            if feature in numerical_features:

                series = pd.to_numeric(
                    training_df[feature],
                    errors="coerce"
                )

                series = series.replace(
                    [np.inf, -np.inf],
                    np.nan
                ).dropna()

                if series.empty:

                    default_value = 0.0

                else:

                    default_value = float(
                        series.median()
                    )

                if not np.isfinite(default_value):

                    default_value = 0.0

                with (
                    left_col
                    if index % 2 == 0
                    else right_col
                ):

                    input_values[feature] = st.number_input(
                        label=feature.replace("_", " "),
                        value=default_value,
                        step=1.0,
                        format="%.2f",
                        key=f"numeric_{feature}"
                    )

            else:

                values = training_df[feature].dropna()

                categories = sorted(
                    values.astype(str).unique().tolist()
                )

                if not categories:

                    categories = ["Unknown"]

                with (
                    left_col
                    if index % 2 == 0
                    else right_col
                ):

                    input_values[feature] = st.selectbox(
                        label=feature.replace("_", " "),
                        options=categories,
                        key=f"category_{feature}"
                    )

        submitted = st.form_submit_button(
            "Predict House Price",
            type="primary",
            use_container_width=True
        )

    if submitted:

        input_df = pd.DataFrame(
            [input_values],
            columns=feature_names
        )

        try:

            prediction = float(
                model.predict(input_df)[0]
            )

            st.markdown("---")

            st.subheader("Prediction Result")

            st.success(
                f"Estimated House Price: {prediction:,.2f}"
            )

            st.caption(
                "The amount is displayed in the same "
                "units as the SalePrice column in your dataset."
            )

            st.subheader("Entered Property Details")

            st.dataframe(
                input_df,
                use_container_width=True
            )

        except Exception as exc:

            st.error(
                f"Prediction could not be generated: {exc}"
            )