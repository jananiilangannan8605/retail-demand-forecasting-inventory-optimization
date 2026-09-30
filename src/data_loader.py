from google.cloud import bigquery

from bigquery_client import get_bigquery_client, get_table_reference


def load_table(table_name, limit=None):
    """
    Load data from a BigQuery table into a pandas DataFrame.
    """

    client = get_bigquery_client()
    table = get_table_reference(table_name)

    query = f"SELECT * FROM `{table}`"

    if limit:
        query += f" LIMIT {limit}"

    return client.query(query).to_dataframe()


def get_table_row_count(table_name):
    """
    Return the number of rows in a BigQuery table.
    """

    client = get_bigquery_client()
    table = get_table_reference(table_name)

    query = f"""
        SELECT COUNT(*) AS row_count
        FROM `{table}`
    """

    result = client.query(query).result()
    row = next(result)

    return row.row_count


if __name__ == "__main__":
    table_name = "calendar"

    row_count = get_table_row_count(table_name)

    print(f"Table: {table_name}")
    print(f"Rows: {row_count}")

    df = load_table(table_name, limit=5)

    print("\nSample data:")
    print(df)