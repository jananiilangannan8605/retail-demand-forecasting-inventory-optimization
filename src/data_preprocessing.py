import pandas as pd
from pathlib import Path


DATA_DIR = Path("data")


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    features = pd.read_csv(DATA_DIR / "features.csv")
    stores = pd.read_csv(DATA_DIR / "stores.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    return train, features, stores, test


def merge_data(train, test, features, stores):
    train["Date"] = pd.to_datetime(train["Date"])
    test["Date"] = pd.to_datetime(test["Date"])
    features["Date"] = pd.to_datetime(features["Date"])

    train = train.merge(
        features,
        on=["Store", "Date", "IsHoliday"],
        how="left"
    )

    test = test.merge(
        features,
        on=["Store", "Date", "IsHoliday"],
        how="left"
    )

    train = train.merge(stores, on="Store", how="left")
    test = test.merge(stores, on="Store", how="left")

    return train, test


def clean_data(df):
    df = df.copy()

    markdown_columns = [
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5"
    ]

    for column in markdown_columns:
        df[column] = df[column].fillna(0)

    for column in ["CPI", "Unemployment"]:
        df[column] = df.groupby("Store")[column].transform(
            lambda x: x.ffill().bfill()
        )
        df[column] = df[column].fillna(df[column].median())

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter
    df["Is_Holiday_Numeric"] = df["IsHoliday"].astype(int)

    return df


def main():
    print("Loading datasets...")

    train, features, stores, test = load_data()

    print(f"Raw train shape: {train.shape}")
    print(f"Raw test shape: {test.shape}")

    print("\nMerging datasets...")

    train, test = merge_data(
        train,
        test,
        features,
        stores
    )

    print(f"Merged train shape: {train.shape}")
    print(f"Merged test shape: {test.shape}")

    print("\nCleaning data...")

    train = clean_data(train)
    test = clean_data(test)

    train.to_csv(
        DATA_DIR / "processed_train.csv",
        index=False
    )

    test.to_csv(
        DATA_DIR / "processed_test.csv",
        index=False
    )

    print("\nProcessing completed successfully!")
    print(f"Processed train shape: {train.shape}")
    print(f"Processed test shape: {test.shape}")

    print("\nCreated:")
    print("data/processed_train.csv")
    print("data/processed_test.csv")


if __name__ == "__main__":
    main()
