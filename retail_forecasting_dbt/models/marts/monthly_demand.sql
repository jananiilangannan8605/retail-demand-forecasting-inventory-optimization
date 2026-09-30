{{ config(materialized='table') }}

SELECT
    DATE_TRUNC(date, MONTH) AS month_start,
    SUM(total_demand) AS total_demand
FROM {{ ref('daily_demand') }}
GROUP BY
    month_start
ORDER BY
    month_start