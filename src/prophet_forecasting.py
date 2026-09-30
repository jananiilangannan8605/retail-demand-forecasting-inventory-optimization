from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ID = "blissful-axiom-509914-p6"
DATASET_ID = "dbt_retail"
TABLE_ID = "prophet_training"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "prophet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_training_data():
    client = bigquery.Client(project=PROJECT_ID)

    query = f"""
        SELECT
            ds,
            y,
            store_id,
            item_id
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
        ORDER BY store_id, item_id, ds
    """

    return client.query(query).to_dataframe()


def train_and_forecast(series_data):
    store_id = series_data["store_id"].iloc[0]
    item_id = series_data["item_id"].iloc[0]

    train = series_data[["ds", "y"]].copy()
    train["ds"] = pd.to_datetime(train["ds"])

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
    )

    model.fit(train)

    future = model.make_future_dataframe(periods=30, freq="D")
    forecast = model.predict(future)

    predictions = forecast[
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].tail(30).copy()

    predictions["store_id"] = store_id
    predictions["item_id"] = item_id

    return predictions


def load_validation_data():
    client = bigquery.Client(project=PROJECT_ID)

    query = f"""
        SELECT
            date AS ds,
            demand AS actual_demand,
            store_id,
            item_id
        FROM `{PROJECT_ID}.dbt_retail.forecast_split`
        WHERE dataset_type = 'validation'
          AND (
                (store_id = 'CA_3' AND item_id = 'FOODS_3_090')
                OR
                (store_id = 'TX_2' AND item_id = 'FOODS_3_586')
                OR
                (store_id = 'TX_3' AND item_id = 'FOODS_3_586')
                OR
                (store_id = 'CA_3' AND item_id = 'FOODS_3_586')
                OR
                (store_id = 'CA_1' AND item_id = 'FOODS_3_090')
              )
        ORDER BY store_id, item_id, ds
    """

    return client.query(query).to_dataframe()


def main():
    print("Loading Prophet training data...")
    training_data = load_training_data()

    print(f"Training rows: {len(training_data):,}")
    print(
        f"Series count: "
        f"{training_data[['store_id', 'item_id']].drop_duplicates().shape[0]}"
    )

    all_predictions = []

    for (store_id, item_id), series_data in training_data.groupby(
        ["store_id", "item_id"]
    ):
        print(f"Training Prophet: {store_id} / {item_id}")

        predictions = train_and_forecast(series_data)
        all_predictions.append(predictions)

    predictions = pd.concat(all_predictions, ignore_index=True)

    print("Loading validation data...")
    validation = load_validation_data()

    predictions["ds"] = pd.to_datetime(predictions["ds"])
    validation["ds"] = pd.to_datetime(validation["ds"])

    results = predictions.merge(
        validation,
        on=["ds", "store_id", "item_id"],
        how="inner",
    )

    results["yhat"] = results["yhat"].clip(lower=0)

    metrics = []

    for (store_id, item_id), group in results.groupby(
        ["store_id", "item_id"]
    ):
        mae = mean_absolute_error(
            group["actual_demand"],
            group["yhat"],
        )

        rmse = np.sqrt(
            mean_squared_error(
                group["actual_demand"],
                group["yhat"],
            )
        )

        metrics.append(
            {
                "store_id": store_id,
                "item_id": item_id,
                "mae": mae,
                "rmse": rmse,
            }
        )

    metrics_df = pd.DataFrame(metrics)

    predictions_path = OUTPUT_DIR / "prophet_predictions.csv"
    metrics_path = OUTPUT_DIR / "prophet_metrics.csv"

    results.to_csv(predictions_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    print("\nProphet forecasting completed.")
    print(f"Predictions saved to: {predictions_path}")
    print(f"Metrics saved to: {metrics_path}")

    print("\nMetrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()