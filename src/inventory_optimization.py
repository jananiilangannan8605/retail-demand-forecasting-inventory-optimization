import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Inventory Planning Assumptions
# =========================================================
# These are configurable business assumptions.
LEAD_TIME_WEEKS = 2
SERVICE_LEVEL_Z = 1.645       # Approx. 95% service level
REVIEW_PERIOD_WEEKS = 1


# =========================================================
# Load Data
# =========================================================
print("=" * 70)
print("DAY 6 - INVENTORY OPTIMIZATION")
print("=" * 70)

train_path = DATA_DIR / "featured_train.csv"
test_path = DATA_DIR / "featured_test.csv"
prediction_path = OUTPUT_DIR / "test_predictions.csv"

print("\nLoading datasets...")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
predictions = pd.read_csv(prediction_path)

train["Date"] = pd.to_datetime(train["Date"])
test["Date"] = pd.to_datetime(test["Date"])

print(f"Historical train data: {train.shape}")
print(f"Forecast test data:    {test.shape}")
print(f"Predictions:            {predictions.shape}")


# =========================================================
# Validate Prediction Data
# =========================================================
if len(test) != len(predictions):
    raise ValueError(
        f"Prediction row count ({len(predictions)}) does not match "
        f"test row count ({len(test)})."
    )

if "Weekly_Sales" not in predictions.columns:
    raise ValueError(
        "The predictions file must contain a 'Weekly_Sales' column."
    )


# =========================================================
# Attach Predictions to Test Data
# =========================================================
forecast = test[
    ["Store", "Dept", "Date"]
].copy()

forecast["Forecast_Sales"] = predictions["Weekly_Sales"].values

forecast["Forecast_Sales"] = (
    forecast["Forecast_Sales"]
    .clip(lower=0)
)

print("\nForecast data prepared successfully.")


# =========================================================
# Historical Demand Statistics
# =========================================================
print("\nCalculating historical demand statistics...")

historical_stats = (
    train.groupby(["Store", "Dept"])["Weekly_Sales"]
    .agg(
        Historical_Avg_Sales="mean",
        Historical_Std_Sales="std",
        Historical_Min_Sales="min",
        Historical_Max_Sales="max",
        Historical_Weeks="count"
    )
    .reset_index()
)

historical_stats["Historical_Std_Sales"] = (
    historical_stats["Historical_Std_Sales"]
    .fillna(0)
)


# =========================================================
# Forecast Statistics
# =========================================================
forecast_stats = (
    forecast.groupby(["Store", "Dept"])["Forecast_Sales"]
    .agg(
        Forecast_Avg_Sales="mean",
        Forecast_Total_Sales="sum",
        Forecast_Max_Sales="max",
        Forecast_Weeks="count"
    )
    .reset_index()
)


# =========================================================
# Merge Historical and Forecast Statistics
# =========================================================
inventory = historical_stats.merge(
    forecast_stats,
    on=["Store", "Dept"],
    how="left"
)


# =========================================================
# Safety Stock Calculation
# =========================================================
# Safety Stock = Z × demand standard deviation × sqrt(lead time)
inventory["Safety_Stock"] = (
    SERVICE_LEVEL_Z
    * inventory["Historical_Std_Sales"]
    * np.sqrt(LEAD_TIME_WEEKS)
)


# =========================================================
# Lead-Time Demand
# =========================================================
inventory["Lead_Time_Demand"] = (
    inventory["Forecast_Avg_Sales"]
    * LEAD_TIME_WEEKS
)


# =========================================================
# Reorder Point
# =========================================================
inventory["Reorder_Point"] = (
    inventory["Lead_Time_Demand"]
    + inventory["Safety_Stock"]
)


# =========================================================
# Recommended Inventory Level
# =========================================================
inventory["Review_Period_Demand"] = (
    inventory["Forecast_Avg_Sales"]
    * REVIEW_PERIOD_WEEKS
)

inventory["Recommended_Inventory"] = (
    inventory["Reorder_Point"]
    + inventory["Review_Period_Demand"]
)


# =========================================================
# Inventory Coverage
# =========================================================
inventory["Safety_Stock_Weeks"] = np.where(
    inventory["Forecast_Avg_Sales"] > 0,
    inventory["Safety_Stock"]
    / inventory["Forecast_Avg_Sales"],
    0
)

inventory["Recommended_Inventory"] = (
    inventory["Recommended_Inventory"]
    .clip(lower=0)
)


# =========================================================
# Round Numeric Values
# =========================================================
numeric_columns = [
    "Historical_Avg_Sales",
    "Historical_Std_Sales",
    "Historical_Min_Sales",
    "Historical_Max_Sales",
    "Forecast_Avg_Sales",
    "Forecast_Total_Sales",
    "Forecast_Max_Sales",
    "Safety_Stock",
    "Lead_Time_Demand",
    "Reorder_Point",
    "Review_Period_Demand",
    "Recommended_Inventory",
    "Safety_Stock_Weeks"
]

for column in numeric_columns:
    inventory[column] = inventory[column].round(2)


# =========================================================
# Inventory Priority
# =========================================================
inventory["Inventory_Priority"] = pd.cut(
    inventory["Recommended_Inventory"],
    bins=[
        -np.inf,
        inventory["Recommended_Inventory"].quantile(0.50),
        inventory["Recommended_Inventory"].quantile(0.90),
        np.inf
    ],
    labels=[
        "Normal",
        "High",
        "Critical"
    ],
    duplicates="drop"
)


# =========================================================
# Save Detailed Inventory Report
# =========================================================
inventory_output = (
    OUTPUT_DIR / "inventory_optimization_report.csv"
)

inventory.to_csv(
    inventory_output,
    index=False
)


# =========================================================
# Store-Level Summary
# =========================================================
store_summary = (
    inventory.groupby("Store")
    .agg(
        Departments=("Dept", "nunique"),
        Forecast_Total_Sales=("Forecast_Total_Sales", "sum"),
        Average_Reorder_Point=("Reorder_Point", "mean"),
        Total_Safety_Stock=("Safety_Stock", "sum"),
        Total_Recommended_Inventory=(
            "Recommended_Inventory",
            "sum"
        )
    )
    .reset_index()
)

store_summary = store_summary.round(2)

store_output = OUTPUT_DIR / "store_inventory_summary.csv"

store_summary.to_csv(
    store_output,
    index=False
)


# =========================================================
# Top Inventory Requirements
# =========================================================
top_inventory = (
    inventory.sort_values(
        "Recommended_Inventory",
        ascending=False
    )
    .head(20)
)

top_inventory_output = (
    OUTPUT_DIR / "top_inventory_requirements.csv"
)

top_inventory.to_csv(
    top_inventory_output,
    index=False
)


# =========================================================
# Summary Statistics
# =========================================================
total_forecast = inventory["Forecast_Total_Sales"].sum()
total_safety_stock = inventory["Safety_Stock"].sum()
total_reorder_point = inventory["Reorder_Point"].sum()
total_recommended = inventory["Recommended_Inventory"].sum()


# =========================================================
# Print Results
# =========================================================
print("\n" + "=" * 70)
print("INVENTORY OPTIMIZATION COMPLETED")
print("=" * 70)

print(f"\nStore-Department combinations: {len(inventory):,}")

print(
    f"Total forecast demand: "
    f"{total_forecast:,.2f}"
)

print(
    f"Total safety stock: "
    f"{total_safety_stock:,.2f}"
)

print(
    f"Total reorder-point quantity: "
    f"{total_reorder_point:,.2f}"
)

print(
    f"Total recommended inventory: "
    f"{total_recommended:,.2f}"
)

print(f"\nLead time assumption: {LEAD_TIME_WEEKS} weeks")
print(f"Service-level Z value: {SERVICE_LEVEL_Z}")
print(f"Review period: {REVIEW_PERIOD_WEEKS} week")

print("\nGenerated files:")
print(f"  - {inventory_output}")
print(f"  - {store_output}")
print(f"  - {top_inventory_output}")

print("\nTop 10 inventory requirements:")

display_columns = [
    "Store",
    "Dept",
    "Forecast_Avg_Sales",
    "Safety_Stock",
    "Reorder_Point",
    "Recommended_Inventory"
]

print(
    top_inventory[display_columns]
    .head(10)
    .to_string(index=False)
)

print("\nDay 6 inventory optimization pipeline finished successfully!")