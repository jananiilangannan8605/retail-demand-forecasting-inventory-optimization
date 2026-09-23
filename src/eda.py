import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
FIGURES_DIR = OUTPUT_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load the processed training dataset."""
    df = pd.read_csv(DATA_DIR / "processed_train.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def basic_analysis(df):
    """Print basic dataset information."""
    print("=" * 60)
    print("BASIC DATASET INFORMATION")
    print("=" * 60)

    print(f"Shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    print("\nDate range:")
    print(f"From: {df['Date'].min().date()}")
    print(f"To:   {df['Date'].max().date()}")

    print("\nNumber of stores:", df["Store"].nunique())
    print("Number of departments:", df["Dept"].nunique())


def sales_statistics(df):
    """Display weekly sales statistics."""
    print("\n" + "=" * 60)
    print("WEEKLY SALES STATISTICS")
    print("=" * 60)

    print(df["Weekly_Sales"].describe())

    print("\nTotal sales:", round(df["Weekly_Sales"].sum(), 2))
    print("Average weekly sales:", round(df["Weekly_Sales"].mean(), 2))


def sales_by_store(df):
    """Calculate and display sales by store."""
    store_sales = (
        df.groupby("Store")["Weekly_Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\n" + "=" * 60)
    print("TOP 10 STORES BY TOTAL SALES")
    print("=" * 60)
    print(store_sales.head(10))

    plt.figure(figsize=(10, 6))
    store_sales.head(10).sort_values().plot(kind="barh")
    plt.title("Top 10 Stores by Total Sales")
    plt.xlabel("Total Weekly Sales")
    plt.ylabel("Store")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_10_stores.png")
    plt.close()


def sales_by_department(df):
    """Calculate and display sales by department."""
    dept_sales = (
        df.groupby("Dept")["Weekly_Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\n" + "=" * 60)
    print("TOP 10 DEPARTMENTS BY TOTAL SALES")
    print("=" * 60)
    print(dept_sales.head(10))

    plt.figure(figsize=(10, 6))
    dept_sales.head(10).sort_values().plot(kind="barh")
    plt.title("Top 10 Departments by Total Sales")
    plt.xlabel("Total Weekly Sales")
    plt.ylabel("Department")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_10_departments.png")
    plt.close()


def monthly_sales_trend(df):
    """Analyze monthly sales trends."""
    monthly_sales = (
        df.groupby(["Year", "Month"])["Weekly_Sales"]
        .sum()
        .reset_index()
    )

    monthly_sales["Period"] = (
        monthly_sales["Year"].astype(str)
        + "-"
        + monthly_sales["Month"].astype(str).str.zfill(2)
    )

    plt.figure(figsize=(14, 6))
    plt.plot(monthly_sales["Period"], monthly_sales["Weekly_Sales"])
    plt.title("Monthly Sales Trend")
    plt.xlabel("Period")
    plt.ylabel("Total Weekly Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "monthly_sales_trend.png")
    plt.close()

    print("\n" + "=" * 60)
    print("MONTHLY SALES TREND")
    print("=" * 60)
    print(monthly_sales.head(12))


def holiday_analysis(df):
    """Compare holiday and non-holiday sales."""
    holiday_sales = (
        df.groupby("IsHoliday")["Weekly_Sales"]
        .agg(["count", "mean", "sum"])
    )

    print("\n" + "=" * 60)
    print("HOLIDAY VS NON-HOLIDAY SALES")
    print("=" * 60)
    print(holiday_sales)

    plt.figure(figsize=(8, 5))
    holiday_sales["mean"].plot(kind="bar")
    plt.title("Average Weekly Sales: Holiday vs Non-Holiday")
    plt.xlabel("Holiday")
    plt.ylabel("Average Weekly Sales")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "holiday_sales_comparison.png")
    plt.close()


def correlation_analysis(df):
    """Analyze correlations between numerical variables."""
    columns = [
        "Weekly_Sales",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5",
    ]

    correlation = df[columns].corr()

    print("\n" + "=" * 60)
    print("CORRELATION WITH WEEKLY SALES")
    print("=" * 60)

    print(
        correlation["Weekly_Sales"]
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 8))
    plt.imshow(correlation, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(columns)), columns, rotation=90)
    plt.yticks(range(len(columns)), columns)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_matrix.png")
    plt.close()


def main():
    print("Loading processed training data...")
    df = load_data()

    basic_analysis(df)
    sales_statistics(df)
    sales_by_store(df)
    sales_by_department(df)
    monthly_sales_trend(df)
    holiday_analysis(df)
    correlation_analysis(df)

    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Charts saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
