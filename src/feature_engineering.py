import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# ---------------------------------------------------------
# Load processed datasets
# ---------------------------------------------------------
train_path = DATA_DIR / "processed_train.csv"
test_path = DATA_DIR / "processed_test.csv"

print("Loading processed datasets...")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

train["Date"] = pd.to_datetime(train["Date"])
test["Date"] = pd.to_datetime(test["Date"])

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")


# ---------------------------------------------------------
# Calendar / Time Features
# ---------------------------------------------------------
def add_time_features(df):
    df = df.copy()

    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayOfMonth"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)

    df["IsQuarterStart"] = df["Date"].dt.is_quarter_start.astype(int)
    df["IsQuarterEnd"] = df["Date"].dt.is_quarter_end.astype(int)

    df["IsYearStart"] = df["Date"].dt.is_year_start.astype(int)
    df["IsYearEnd"] = df["Date"].dt.is_year_end.astype(int)

    min_date = df["Date"].min()
    df["DaysSinceStart"] = (df["Date"] - min_date).dt.days

    return df


# ---------------------------------------------------------
# Add time features to train and test
# ---------------------------------------------------------
train = add_time_features(train)
test = add_time_features(test)


# ---------------------------------------------------------
# Sort training data
# ---------------------------------------------------------
train = train.sort_values(
    by=["Store", "Dept", "Date"]
).reset_index(drop=True)


# ---------------------------------------------------------
# Lag Features
# ---------------------------------------------------------
group_columns = ["Store", "Dept"]

lag_periods = [1, 2, 4, 13, 26, 52]

print("\nCreating lag features...")

for lag in lag_periods:
    column_name = f"Sales_Lag_{lag}"

    train[column_name] = (
        train.groupby(group_columns)["Weekly_Sales"]
        .shift(lag)
    )

    print(f"Created {column_name}")


# ---------------------------------------------------------
# Rolling Mean Features
# ---------------------------------------------------------
rolling_windows = [4, 13, 26, 52]

print("\nCreating rolling mean features...")

for window in rolling_windows:
    column_name = f"Sales_Rolling_Mean_{window}"

    train[column_name] = (
        train.groupby(group_columns)["Weekly_Sales"]
        .transform(
            lambda x: x.shift(1).rolling(
                window=window,
                min_periods=1
            ).mean()
        )
    )

    print(f"Created {column_name}")


# ---------------------------------------------------------
# Rolling Standard Deviation Features
# ---------------------------------------------------------
rolling_std_windows = [4, 13]

print("\nCreating rolling standard deviation features...")

for window in rolling_std_windows:
    column_name = f"Sales_Rolling_Std_{window}"

    train[column_name] = (
        train.groupby(group_columns)["Weekly_Sales"]
        .transform(
            lambda x: x.shift(1).rolling(
                window=window,
                min_periods=2
            ).std()
        )
    )

    print(f"Created {column_name}")


# ---------------------------------------------------------
# Price / Economic Features
# ---------------------------------------------------------
if "Temperature" in train.columns:
    train["Temperature_Squared"] = train["Temperature"] ** 2
    test["Temperature_Squared"] = test["Temperature"] ** 2

if "Fuel_Price" in train.columns:
    train["Fuel_Price_Squared"] = train["Fuel_Price"] ** 2
    test["Fuel_Price_Squared"] = test["Fuel_Price"] ** 2

if "CPI" in train.columns:
    train["CPI_Squared"] = train["CPI"] ** 2
    test["CPI_Squared"] = test["CPI"] ** 2

if "Unemployment" in train.columns:
    train["Unemployment_Squared"] = train["Unemployment"] ** 2
    test["Unemployment_Squared"] = test["Unemployment"] ** 2


# ---------------------------------------------------------
# Store / Department Interaction
# ---------------------------------------------------------
train["Store_Dept"] = (
    train["Store"].astype(str)
    + "_"
    + train["Dept"].astype(str)
)

test["Store_Dept"] = (
    test["Store"].astype(str)
    + "_"
    + test["Dept"].astype(str)
)


# ---------------------------------------------------------
# Holiday Interaction
# ---------------------------------------------------------
if "Is_Holiday_Numeric" in train.columns:
    train["Holiday_Store_Interaction"] = (
        train["Is_Holiday_Numeric"] * train["Store"]
    )

    test["Holiday_Store_Interaction"] = (
        test["Is_Holiday_Numeric"] * test["Store"]
    )


# ---------------------------------------------------------
# Handle infinite values
# ---------------------------------------------------------
train = train.replace([np.inf, -np.inf], np.nan)
test = test.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------
# Save feature-engineered datasets
# ---------------------------------------------------------
train_output = DATA_DIR / "featured_train.csv"
test_output = DATA_DIR / "featured_test.csv"

train.to_csv(train_output, index=False)
test.to_csv(test_output, index=False)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print(f"\nFeatured train shape: {train.shape}")
print(f"Featured test shape:  {test.shape}")

print(f"\nTrain output: {train_output}")
print(f"Test output:  {test_output}")

print("\nNew feature columns:")

original_columns = [
    "Store",
    "Dept",
    "Date",
    "Weekly_Sales",
    "IsHoliday"
]

new_columns = [
    column for column in train.columns
    if column not in original_columns
]

for column in new_columns:
    print(f"  - {column}")

print("\nFeature engineering pipeline finished successfully!")
