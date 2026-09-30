from google.cloud import bigquery
import pandas as pd

from bigquery_client import get_bigquery_client, get_table_reference


def get_daily_demand():
    """Calculate total daily demand from the M5 validation dataset."""

    client = get_bigquery_client()
    sales_table = get_table_reference("sales_train_validation")
    calendar_table = get_table_reference("calendar")

    # Get all daily sales columns dynamically
    query_columns = f"""
        SELECT column_name
        FROM `{client.project}.retail_forecasting.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = 'sales_train_validation'
          AND STARTS_WITH(column_name, 'd_')
        ORDER BY ordinal_position
    """

    columns = [
        row.column_name
        for row in client.query(query_columns).result()
    ]

    print(f"Daily sales columns found: {len(columns)}")

    # Build UNPIVOT column list dynamically
    unpivot_columns = ", ".join(columns)

    query = f"""
        SELECT
            c.date,
            c.d,
            SUM(s.sales) AS total_demand
        FROM `{sales_table}`
        UNPIVOT (
            sales FOR d IN ({unpivot_columns})
        ) AS s
        JOIN `{calendar_table}` AS c
            ON c.d = s.d
        GROUP BY c.date, c.d
        ORDER BY c.date
    """

    return client.query(query).to_dataframe()


if __name__ == "__main__":
    df = get_daily_demand()

    print("\nDaily demand sample:")
    print(df.head(10))

    print("\nTotal days:", len(df))

    print("\nDemand summary:")
    print(df["total_demand"].describe())