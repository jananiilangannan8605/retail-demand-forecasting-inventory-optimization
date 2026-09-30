{{ config(materialized='table') }}

SELECT
    date,
    d,
    SUM(demand) AS total_demand
FROM {{ ref('stg_sales') }}
GROUP BY
    date,
    d
ORDER BY
    date