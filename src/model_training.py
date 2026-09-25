import argparse
import json
import time
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Feature Definitions
# ---------------------------------------------------------
EXOGENOUS_FEATURES = [
    # Store & Department
    "Store",
    "Dept",
    "Type_Encoded",
    "Size",
    # Economic & Weather
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    # Markdowns
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
    # Calendar & Temporal
    "Month",
    "Week",
    "Quarter",
    "DayOfWeek",
    "DayOfMonth",
    "WeekOfYear",
    "IsMonthStart",
    "IsMonthEnd",
    "IsQuarterStart",
    "IsQuarterEnd",
    "DaysSinceStart",
    "Is_Holiday_Numeric",
    # Interaction Features
    "Holiday_Store_Interaction",
    "Temperature_Squared",
    "Fuel_Price_Squared",
    "CPI_Squared",
    "Unemployment_Squared",
]

LAG_FEATURES = [
    # Sales Lags
    "Sales_Lag_1",
    "Sales_Lag_2",
    "Sales_Lag_4",
    "Sales_Lag_13",
    "Sales_Lag_26",
    "Sales_Lag_52",
    # Rolling Statistics
    "Sales_Rolling_Mean_4",
    "Sales_Rolling_Mean_13",
    "Sales_Rolling_Mean_26",
    "Sales_Rolling_Mean_52",
    "Sales_Rolling_Std_4",
    "Sales_Rolling_Std_13",
]


# ---------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------
def calculate_wmae(y_true, y_pred, is_holiday):
    """
    Weighted Mean Absolute Error (WMAE) as defined in Walmart Sales Forecasting.
    Holiday weeks receive a weight of 5, while non-holiday weeks receive 1.
    """
    weights = np.where(is_holiday, 5.0, 1.0)
    wmae = np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)
    return float(wmae)


def compute_metrics(y_true, y_pred, is_holiday):
    """Calculate regression evaluation metrics: WMAE, MAE, RMSE, and R2."""
    wmae = calculate_wmae(y_true, y_pred, is_holiday)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "WMAE": round(wmae, 2),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
    }


# ---------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------
def prepare_data(df, include_lags=True):
    """Preprocess data and extract available feature columns."""
    df = df.copy()

    # Encode categorical store 'Type'
    if "Type" in df.columns:
        type_mapping = {"A": 1, "B": 2, "C": 3}
        df["Type_Encoded"] = df["Type"].map(type_mapping).fillna(0).astype(int)

    feature_cols = [col for col in EXOGENOUS_FEATURES if col in df.columns]

    if include_lags:
        available_lags = [col for col in LAG_FEATURES if col in df.columns]
        feature_cols.extend(available_lags)

    return df, feature_cols


def temporal_train_val_split(df, split_date="2012-02-01", include_lags=True):
    """
    Split the dataset chronologically to avoid future data leakage.
    Train: data prior to split_date
    Validation: data on or after split_date
    """
    df, feature_cols = prepare_data(df, include_lags=include_lags)

    train_mask = df["Date"] < split_date
    val_mask = df["Date"] >= split_date

    train_df = df[train_mask]
    val_df = df[val_mask]

    X_train = train_df[feature_cols]
    y_train = train_df["Weekly_Sales"]

    X_val = val_df[feature_cols]
    y_val = val_df["Weekly_Sales"]
    val_is_holiday = val_df["Is_Holiday_Numeric"] == 1

    return X_train, y_train, X_val, y_val, val_is_holiday, feature_cols, val_df


# ---------------------------------------------------------
# Model Factory
# ---------------------------------------------------------
def get_models(fast_mode=False):
    """Return dictionary of candidate models to evaluate."""
    if fast_mode:
        models = {
            "Ridge_Regression": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("regressor", Ridge(alpha=10.0, random_state=42)),
                ]
            ),
            "HistGradientBoosting": HistGradientBoostingRegressor(
                max_iter=50,
                max_depth=8,
                random_state=42,
            ),
            "Random_Forest": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "regressor",
                        RandomForestRegressor(
                            n_estimators=30,
                            max_depth=12,
                            n_jobs=-1,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        }
    else:
        models = {
            "Ridge_Regression": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("regressor", Ridge(alpha=1.0, random_state=42)),
                ]
            ),
            "HistGradientBoosting": HistGradientBoostingRegressor(
                max_iter=150,
                learning_rate=0.08,
                max_depth=14,
                min_samples_leaf=20,
                l2_regularization=0.1,
                random_state=42,
            ),
            "Random_Forest": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "regressor",
                        RandomForestRegressor(
                            n_estimators=60,
                            max_depth=18,
                            min_samples_split=5,
                            n_jobs=-1,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        }

    return models


# ---------------------------------------------------------
# Plotting & Diagnostics
# ---------------------------------------------------------
def plot_actual_vs_predicted(val_df, y_true, y_pred, model_name):
    """Plot sample actual vs predicted weekly sales time series."""
    temp_df = val_df[["Date", "Store", "Dept"]].copy()
    temp_df["Actual"] = y_true.values
    temp_df["Predicted"] = y_pred

    # Aggregate weekly overall sales to view global trend
    weekly_agg = temp_df.groupby("Date")[["Actual", "Predicted"]].sum().reset_index()

    plt.figure(figsize=(14, 6))
    plt.plot(
        weekly_agg["Date"],
        weekly_agg["Actual"],
        label="Actual Weekly Sales",
        color="#1f77b4",
        linewidth=2,
        marker="o",
    )
    plt.plot(
        weekly_agg["Date"],
        weekly_agg["Predicted"],
        label=f"Predicted ({model_name})",
        color="#d62728",
        linewidth=2,
        linestyle="--",
        marker="s",
    )
    plt.title(
        f"Validation Set: Total Weekly Sales (Actual vs Predicted) - {model_name}",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Total Weekly Sales ($)", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    save_path = FIGURES_DIR / "validation_actual_vs_predicted.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved prediction plot to: {save_path}")


def save_feature_importance(model, feature_names):
    """Extract and save feature importance plot if model supports it."""
    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.named_steps.get("regressor", model)

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        indices = np.argsort(importances)[::-1][:20]

        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_importances[::-1], color="#2ca02c", align="center")
        plt.yticks(range(len(top_features)), top_features[::-1])
        plt.xlabel("Relative Importance")
        plt.title("Top 20 Feature Importances", fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_path = FIGURES_DIR / "top_feature_importances.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved feature importance plot to: {save_path}")


# ---------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------
def train_pipeline(
    train_path=DATA_DIR / "featured_train.csv",
    split_date="2012-02-01",
    include_lags=True,
    selected_model="all",
    fast_mode=False,
    predict_test=False,
):
    print("=" * 60)
    print("RETAIL DEMAND FORECASTING - MODEL TRAINING PIPELINE")
    print("=" * 60)
    print(f"Train Dataset:       {train_path}")
    print(f"Temporal Split Date: {split_date}")
    print(f"Include Lags:        {include_lags}")
    print(f"Selected Model:      {selected_model}")
    print(f"Fast Mode:           {fast_mode}")

    # 1. Load Data
    print("\nLoading dataset...")
    df = pd.read_csv(train_path)
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"Loaded dataset with shape: {df.shape}")

    # 2. Split Data
    print(f"\nSplitting data chronologically at '{split_date}'...")
    (
        X_train,
        y_train,
        X_val,
        y_val,
        val_is_holiday,
        feature_cols,
        val_df,
    ) = temporal_train_val_split(
        df,
        split_date=split_date,
        include_lags=include_lags,
    )

    print(f"Train samples:      {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Features count:     {len(feature_cols)}")

    # 3. Model Training & Evaluation
    available_models = get_models(fast_mode=fast_mode)

    if selected_model != "all":
        if selected_model in available_models:
            available_models = {selected_model: available_models[selected_model]}
        else:
            raise ValueError(
                f"Unknown model '{selected_model}'. Available: {list(available_models.keys())} or 'all'"
            )

    results = {}
    fitted_models = {}
    val_predictions = {}

    print("\n" + "=" * 60)
    print("TRAINING AND EVALUATING CANDIDATE MODELS")
    print("=" * 60)

    for model_name, model in available_models.items():
        print(f"\nTraining [{model_name}]...")
        start_time = time.time()
        model.fit(X_train, y_train)
        elapsed_sec = round(time.time() - start_time, 2)

        preds = model.predict(X_val)
        val_predictions[model_name] = preds
        fitted_models[model_name] = model

        metrics = compute_metrics(y_val, preds, val_is_holiday)
        metrics["Training_Time_Seconds"] = elapsed_sec
        results[model_name] = metrics

        print(f"-> Completed in {elapsed_sec}s")
        print(
            f"   WMAE: {metrics['WMAE']} | MAE: {metrics['MAE']} | RMSE: {metrics['RMSE']} | R2: {metrics['R2']}"
        )

    # 4. Results Summary
    summary_df = pd.DataFrame(results).T.sort_values(by="WMAE")
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (Ranked by WMAE)")
    print("=" * 60)
    print(summary_df.to_string())

    best_model_name = summary_df.index[0]
    best_model = fitted_models[best_model_name]
    best_wmae = results[best_model_name]["WMAE"]
    print(f"\n>>> Best Model: [{best_model_name}] with WMAE: {best_wmae} <<<")

    # 5. Persist Best Model and Metrics
    best_model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_model_name,
            "feature_cols": feature_cols,
            "metrics": results[best_model_name],
            "include_lags": include_lags,
        },
        best_model_path,
    )
    print(f"\nSaved best model bundle to: {best_model_path}")

    metrics_json_path = OUTPUTS_DIR / "model_evaluation_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(
            {
                "best_model": best_model_name,
                "split_date": split_date,
                "include_lags": include_lags,
                "models_evaluation": results,
            },
            f,
            indent=4,
        )
    print(f"Saved evaluation metrics to:  {metrics_json_path}")

    # 6. Generate Diagnostics Plots
    plot_actual_vs_predicted(
        val_df,
        y_val,
        val_predictions[best_model_name],
        best_model_name,
    )
    save_feature_importance(best_model, feature_cols)

    # 7. Optional Test Set Inference
    if predict_test:
        test_file = DATA_DIR / "featured_test.csv"
        if test_file.exists():
            print(f"\nGenerating predictions for test set: {test_file}...")
            test_df = pd.read_csv(test_file)
            test_df["Date"] = pd.to_datetime(test_df["Date"])
            prepared_test, _ = prepare_data(test_df, include_lags=False)

            # Check if all required features exist
            missing_features = [col for col in feature_cols if col not in prepared_test.columns]
            if missing_features:
                print(
                    f"Warning: Test set missing {len(missing_features)} features (e.g. Lags). "
                    "Imputing missing lag/rolling features using historical Store-Dept sales statistics."
                )
                store_dept_median = (
                    df.groupby(["Store", "Dept"])["Weekly_Sales"]
                    .median()
                    .to_dict()
                )
                global_median = df["Weekly_Sales"].median()
                store_dept_keys = list(zip(prepared_test["Store"], prepared_test["Dept"]))
                imputed_sales = [store_dept_median.get(k, global_median) for k in store_dept_keys]

                for col in missing_features:
                    if "Lag" in col or "Rolling_Mean" in col:
                        prepared_test[col] = imputed_sales
                    else:
                        prepared_test[col] = 0

            test_preds = best_model.predict(prepared_test[feature_cols])

            # Prepare submission format: Store_Dept_Date, Weekly_Sales
            test_df["Weekly_Sales"] = test_preds
            test_df["Id"] = (
                test_df["Store"].astype(str)
                + "_"
                + test_df["Dept"].astype(str)
                + "_"
                + test_df["Date"].dt.strftime("%Y-%m-%d")
            )
            submission_output = OUTPUTS_DIR / "test_predictions.csv"
            test_df[["Id", "Weekly_Sales"]].to_csv(submission_output, index=False)
            print(f"Saved test predictions to: {submission_output}")
        else:
            print(f"\nNote: Test file {test_file} not found; skipping test prediction.")

    print("\n" + "=" * 60)
    print("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train demand forecasting regression models for Walmart retail sales."
    )
    parser.add_argument(
        "--split-date",
        type=str,
        default="2012-02-01",
        help="Date threshold for temporal split (default: 2012-02-01).",
    )
    parser.add_argument(
        "--no-lags",
        dest="include_lags",
        action="store_false",
        help="Exclude lag and rolling sales features (useful for direct exogenous test inference).",
    )
    parser.set_defaults(include_lags=True)
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "HistGradientBoosting", "Random_Forest", "Ridge_Regression"],
        help="Model to train and evaluate (default: all).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable fast mode with smaller trees for rapid iteration.",
    )
    parser.add_argument(
        "--predict-test",
        action="store_true",
        help="Generate predictions on featured_test.csv using the best trained model.",
    )

    args = parser.parse_args()

    train_pipeline(
        split_date=args.split_date,
        include_lags=args.include_lags,
        selected_model=args.model,
        fast_mode=args.fast,
        predict_test=args.predict_test,
    )
