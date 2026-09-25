import argparse
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Evaluation Helper Functions
# ---------------------------------------------------------
def calculate_wmae(y_true, y_pred, is_holiday):
    """
    Weighted Mean Absolute Error (WMAE).
    Holiday weeks receive 5x weight; non-holiday weeks receive 1x weight.
    """
    if np.isscalar(is_holiday):
        return float(mean_absolute_error(y_true, y_pred))
    weights = np.where(is_holiday, 5.0, 1.0)
    return float(np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights))


def calculate_metrics_dict(y_true, y_pred, is_holiday):
    """Compute WMAE, MAE, RMSE, and R2."""
    wmae = calculate_wmae(y_true, y_pred, is_holiday)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    return {
        "WMAE": round(wmae, 2),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "Mean_Bias": round(bias, 2),
    }


# ---------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------
def prepare_validation_data(df, feature_cols, split_date="2012-02-01"):
    """Prepare out-of-time validation slice matching model training."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    if "Type" in df.columns:
        type_mapping = {"A": 1, "B": 2, "C": 3}
        df["Type_Encoded"] = df["Type"].map(type_mapping).fillna(0).astype(int)

    val_df = df[df["Date"] >= split_date].copy()

    # Fill any missing feature columns with 0
    for col in feature_cols:
        if col not in val_df.columns:
            val_df[col] = 0.0

    X_val = val_df[feature_cols]
    y_val = val_df["Weekly_Sales"]
    val_is_holiday = val_df["Is_Holiday_Numeric"] == 1

    return val_df, X_val, y_val, val_is_holiday


# ---------------------------------------------------------
# Analysis Functions
# ---------------------------------------------------------
def residual_statistics_analysis(val_df, y_val, y_pred):
    """Analyze overall residual distribution, bias, and error tolerance bands."""
    residuals = y_val - y_pred
    abs_errors = np.abs(residuals)

    # Safe percentage error calculation (avoid division by 0)
    non_zero_mask = y_val > 10.0
    percentage_errors = (
        np.abs(residuals[non_zero_mask]) / y_val[non_zero_mask]
    ) * 100.0

    within_10_pct = float(np.mean(percentage_errors <= 10.0) * 100.0)
    within_20_pct = float(np.mean(percentage_errors <= 20.0) * 100.0)
    within_30_pct = float(np.mean(percentage_errors <= 30.0) * 100.0)

    stats = {
        "Total_Observations": int(len(residuals)),
        "Mean_Residual_Bias": round(float(residuals.mean()), 2),
        "Median_Residual": round(float(residuals.median()), 2),
        "Std_Residual": round(float(residuals.std()), 2),
        "Skewness": round(float(residuals.skew()), 4),
        "Kurtosis": round(float(residuals.kurt()), 4),
        "Max_Underprediction": round(float(residuals.max()), 2),
        "Max_Overprediction": round(float(abs(residuals.min())), 2),
        "Median_Absolute_Error": round(float(abs_errors.median()), 2),
        "Pct_Predictions_Within_10pct": round(within_10_pct, 2),
        "Pct_Predictions_Within_20pct": round(within_20_pct, 2),
        "Pct_Predictions_Within_30pct": round(within_30_pct, 2),
    }

    print("\n" + "=" * 60)
    print("RESIDUAL & ERROR DISTRIBUTION STATISTICS")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:30s}: {v}")

    return stats


def holiday_vs_non_holiday_analysis(val_df, y_val, y_pred):
    """Compare forecasting accuracy between holiday and regular weeks."""
    val_df = val_df.copy()
    val_df["Actual"] = y_val
    val_df["Predicted"] = y_pred
    val_df["Abs_Error"] = np.abs(y_val - y_pred)

    holiday_mask = val_df["Is_Holiday_Numeric"] == 1
    non_holiday_mask = ~holiday_mask

    holiday_metrics = calculate_metrics_dict(
        y_val[holiday_mask],
        y_pred[holiday_mask],
        is_holiday=True,
    )
    non_holiday_metrics = calculate_metrics_dict(
        y_val[non_holiday_mask],
        y_pred[non_holiday_mask],
        is_holiday=False,
    )

    summary = {
        "Holiday_Weeks": {
            "Count": int(holiday_mask.sum()),
            **holiday_metrics,
        },
        "Non_Holiday_Weeks": {
            "Count": int(non_holiday_mask.sum()),
            **non_holiday_metrics,
        },
    }

    print("\n" + "=" * 60)
    print("HOLIDAY VS NON-HOLIDAY PERFORMANCE BREAKDOWN")
    print("=" * 60)
    print(pd.DataFrame(summary).T.to_string())

    return summary


def store_and_dept_breakdown(val_df, y_val, y_pred, top_n=10):
    """Identify stores and departments with the highest and lowest forecast errors."""
    val_df = val_df.copy()
    val_df["Actual"] = y_val
    val_df["Predicted"] = y_pred
    val_df["Abs_Error"] = np.abs(y_val - y_pred)
    val_df["Squared_Error"] = (y_val - y_pred) ** 2
    val_df["Weight"] = np.where(val_df["Is_Holiday_Numeric"] == 1, 5.0, 1.0)
    val_df["Weighted_Abs_Error"] = val_df["Weight"] * val_df["Abs_Error"]

    # Store-level aggregation
    store_summary = val_df.groupby("Store").apply(
        lambda g: pd.Series(
            {
                "Total_Sales": g["Actual"].sum(),
                "MAE": g["Abs_Error"].mean(),
                "RMSE": np.sqrt(g["Squared_Error"].mean()),
                "WMAE": g["Weighted_Abs_Error"].sum() / g["Weight"].sum(),
                "Observations": len(g),
            }
        ),
        include_groups=False,
    )

    # Department-level aggregation
    dept_summary = val_df.groupby("Dept").apply(
        lambda g: pd.Series(
            {
                "Total_Sales": g["Actual"].sum(),
                "MAE": g["Abs_Error"].mean(),
                "RMSE": np.sqrt(g["Squared_Error"].mean()),
                "WMAE": g["Weighted_Abs_Error"].sum() / g["Weight"].sum(),
                "Observations": len(g),
            }
        ),
        include_groups=False,
    )

    # Store Type aggregation
    type_summary = val_df.groupby("Type").apply(
        lambda g: pd.Series(
            {
                "Total_Sales": g["Actual"].sum(),
                "MAE": g["Abs_Error"].mean(),
                "RMSE": np.sqrt(g["Squared_Error"].mean()),
                "WMAE": g["Weighted_Abs_Error"].sum() / g["Weight"].sum(),
                "Observations": len(g),
            }
        ),
        include_groups=False,
    )

    top_error_stores = store_summary.sort_values(by="WMAE", ascending=False).head(top_n)
    top_error_depts = dept_summary.sort_values(by="WMAE", ascending=False).head(top_n)

    print("\n" + "=" * 60)
    print(f"TOP {top_n} HIGHEST ERROR DEPARTMENTS (By WMAE)")
    print("=" * 60)
    print(top_error_depts[["MAE", "RMSE", "WMAE", "Total_Sales", "Observations"]].round(2).to_string())

    print("\n" + "=" * 60)
    print(f"TOP {top_n} HIGHEST ERROR STORES (By WMAE)")
    print("=" * 60)
    print(top_error_stores[["MAE", "RMSE", "WMAE", "Total_Sales", "Observations"]].round(2).to_string())

    print("\n" + "=" * 60)
    print("PERFORMANCE BY STORE TYPE")
    print("=" * 60)
    print(type_summary[["MAE", "RMSE", "WMAE", "Total_Sales", "Observations"]].round(2).to_string())

    return {
        "store_summary": store_summary,
        "dept_summary": dept_summary,
        "type_summary": type_summary,
        "top_error_stores": top_error_stores,
        "top_error_depts": top_error_depts,
    }


def temporal_error_analysis(val_df, y_val, y_pred):
    """Analyze monthly error progression over the validation window."""
    val_df = val_df.copy()
    val_df["Actual"] = y_val
    val_df["Predicted"] = y_pred
    val_df["Abs_Error"] = np.abs(y_val - y_pred)
    val_df["Year_Month"] = val_df["Date"].dt.to_period("M").astype(str)
    val_df["Weight"] = np.where(val_df["Is_Holiday_Numeric"] == 1, 5.0, 1.0)
    val_df["Weighted_Abs_Error"] = val_df["Weight"] * val_df["Abs_Error"]

    monthly = val_df.groupby("Year_Month").apply(
        lambda g: pd.Series(
            {
                "MAE": g["Abs_Error"].mean(),
                "WMAE": g["Weighted_Abs_Error"].sum() / g["Weight"].sum(),
                "Total_Sales": g["Actual"].sum(),
                "Count": len(g),
            }
        ),
        include_groups=False,
    )

    print("\n" + "=" * 60)
    print("MONTHLY VALIDATION ERROR PROGRESSION")
    print("=" * 60)
    print(monthly.round(2).to_string())

    return monthly


# ---------------------------------------------------------
# Plotting & Diagnostic Visualizations
# ---------------------------------------------------------
def generate_diagnostic_plots(val_df, y_val, y_pred, breakdown_results, monthly_results):
    """Generate comprehensive diagnostic figures."""
    residuals = y_val - y_pred

    # 1. Residual Analysis (Histogram & Scatter)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Error distribution
    axes[0].hist(
        residuals,
        bins=80,
        range=(-10000, 10000),
        color="#2b5c8f",
        edgecolor="black",
        alpha=0.7,
    )
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Bias")
    axes[0].set_title("Residual Error Distribution (Actual - Predicted)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Residual ($)", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # Residuals vs Predicted
    sample_indices = np.random.RandomState(42).choice(len(y_pred), size=min(10000, len(y_pred)), replace=False)
    axes[1].scatter(
        y_pred.iloc[sample_indices],
        residuals.iloc[sample_indices],
        alpha=0.2,
        color="#e76f51",
        s=10,
    )
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.5)
    axes[1].set_title("Residuals vs Predicted Sales (Sampled)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted Weekly Sales ($)", fontsize=11)
    axes[1].set_ylabel("Residual ($)", fontsize=11)
    axes[1].set_ylim(-15000, 15000)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    res_path = FIGURES_DIR / "residual_analysis.png"
    plt.savefig(res_path, dpi=300)
    plt.close()
    print(f"Saved: {res_path}")

    # 2. Top Error Departments
    top_depts = breakdown_results["top_error_depts"]
    plt.figure(figsize=(10, 6))
    bars = plt.barh(
        [f"Dept {int(d)}" for d in top_depts.index[::-1]],
        top_depts["WMAE"].values[::-1],
        color="#d95f02",
    )
    plt.title("Top 10 Hardest-to-Forecast Departments (by WMAE)", fontsize=14, fontweight="bold")
    plt.xlabel("Weighted Mean Absolute Error (WMAE in $)", fontsize=12)
    plt.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    dept_path = FIGURES_DIR / "top_error_departments.png"
    plt.savefig(dept_path, dpi=300)
    plt.close()
    print(f"Saved: {dept_path}")

    # 3. Monthly Error Progression
    plt.figure(figsize=(12, 5))
    plt.plot(
        monthly_results.index,
        monthly_results["WMAE"],
        marker="o",
        linewidth=2,
        color="#7570b3",
        label="WMAE",
    )
    plt.plot(
        monthly_results.index,
        monthly_results["MAE"],
        marker="s",
        linewidth=2,
        linestyle="--",
        color="#1b9e77",
        label="MAE",
    )
    plt.title("Validation Error Progression Across 2012", fontsize=14, fontweight="bold")
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Error ($)", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    month_path = FIGURES_DIR / "monthly_validation_error_trend.png"
    plt.savefig(month_path, dpi=300)
    plt.close()
    print(f"Saved: {month_path}")


# ---------------------------------------------------------
# Main Analysis Pipeline
# ---------------------------------------------------------
def run_model_analysis(
    model_path=MODELS_DIR / "best_model.joblib",
    data_path=DATA_DIR / "featured_train.csv",
    split_date="2012-02-01",
    top_n=10,
):
    print("=" * 60)
    print("RETAIL DEMAND FORECASTING - DEEP MODEL ERROR ANALYSIS")
    print("=" * 60)
    print(f"Model Bundle:    {model_path}")
    print(f"Dataset:         {data_path}")
    print(f"Validation From: {split_date}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found at {model_path}. Train the model first!")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}!")

    # 1. Load Model Bundle
    print("\nLoading model bundle...")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    model_name = bundle.get("model_name", "Best_Model")
    feature_cols = bundle["feature_cols"]
    baseline_metrics = bundle.get("metrics", {})
    print(f"Loaded: [{model_name}] trained with {len(feature_cols)} features.")

    # 2. Load and Prepare Validation Data
    print(f"\nLoading dataset and isolating validation slice (>={split_date})...")
    df = pd.read_csv(data_path)
    val_df, X_val, y_val, val_is_holiday = prepare_validation_data(
        df,
        feature_cols=feature_cols,
        split_date=split_date,
    )
    print(f"Validation instances: {len(X_val):,}")

    # 3. Generate Predictions
    print("\nGenerating model predictions on validation set...")
    y_pred = pd.Series(model.predict(X_val), index=y_val.index)

    # Overall Metrics
    overall_metrics = calculate_metrics_dict(y_val, y_pred, val_is_holiday)
    print("\n" + "=" * 60)
    print(f"OVERALL VALIDATION METRICS - [{model_name}]")
    print("=" * 60)
    for k, v in overall_metrics.items():
        print(f"  {k:15s}: {v}")

    # 4. Statistical Residual Analysis
    residual_stats = residual_statistics_analysis(val_df, y_val, y_pred)

    # 5. Holiday vs Non-Holiday Breakdown
    holiday_summary = holiday_vs_non_holiday_analysis(val_df, y_val, y_pred)

    # 6. Store and Department Level Breakdown
    breakdown_results = store_and_dept_breakdown(val_df, y_val, y_pred, top_n=top_n)

    # 7. Temporal Monthly Error Tracking
    monthly_results = temporal_error_analysis(val_df, y_val, y_pred)

    # 8. Generate Diagnostic Plots
    print("\nGenerating diagnostic visualization plots...")
    generate_diagnostic_plots(val_df, y_val, y_pred, breakdown_results, monthly_results)

    # 9. Save Structured Results
    analysis_output = {
        "model_name": model_name,
        "split_date": split_date,
        "overall_metrics": overall_metrics,
        "baseline_training_metrics": baseline_metrics,
        "residual_statistics": residual_stats,
        "holiday_analysis": holiday_summary,
        "top_error_departments": breakdown_results["top_error_depts"]["WMAE"].to_dict(),
        "top_error_stores": breakdown_results["top_error_stores"]["WMAE"].to_dict(),
        "store_type_performance": breakdown_results["type_summary"].round(2).to_dict(orient="index"),
        "monthly_error_progression": monthly_results.round(2).to_dict(orient="index"),
    }

    json_path = OUTPUTS_DIR / "model_error_analysis.json"
    with open(json_path, "w") as f:
        json.dump(analysis_output, f, indent=4)
    print(f"\nSaved structured analysis report to: {json_path}")

    # Save Store and Department Error CSV
    store_csv_path = OUTPUTS_DIR / "store_error_breakdown.csv"
    dept_csv_path = OUTPUTS_DIR / "department_error_breakdown.csv"
    breakdown_results["store_summary"].round(2).to_csv(store_csv_path)
    breakdown_results["dept_summary"].round(2).to_csv(dept_csv_path)
    print(f"Saved store error breakdown to:      {store_csv_path}")
    print(f"Saved department error breakdown to: {dept_csv_path}")

    print("\n" + "=" * 60)
    print("MODEL ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


# ---------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive diagnostic analysis and error breakdown for retail demand models."
    )
    parser.add_argument(
        "--split-date",
        type=str,
        default="2012-02-01",
        help="Chronological validation split date (default: 2012-02-01).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODELS_DIR / "best_model.joblib"),
        help="Path to trained model bundle joblib file.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top error stores and departments to report (default: 10).",
    )

    args = parser.parse_args()

    run_model_analysis(
        model_path=Path(args.model_path),
        split_date=args.split_date,
        top_n=args.top_n,
    )
