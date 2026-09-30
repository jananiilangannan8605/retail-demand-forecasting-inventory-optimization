from google.cloud import bigquery

PROJECT_ID = "blissful-axiom-509914-p6"
DATASET_ID = "retail_forecasting"


def get_bigquery_client():
    """Create and return a BigQuery client."""
    return bigquery.Client(project=PROJECT_ID)


def get_table_reference(table_name):
    """Return the fully qualified BigQuery table name."""
    return f"{PROJECT_ID}.{DATASET_ID}.{table_name}"


if __name__ == "__main__":
    client = get_bigquery_client()

    query = f"""
        SELECT COUNT(*) AS row_count
        FROM `{get_table_reference("calendar")}`
    """

    result = client.query(query).result()
    row = next(result)

    print("BigQuery connection successful")
    print(f"Project: {PROJECT_ID}")
    print(f"Calendar rows: {row.row_count}")