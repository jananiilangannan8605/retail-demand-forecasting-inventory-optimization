{{ config(materialized='table') }}

SELECT
    DATE_TRUNC(date, WEEK(SUNDAY)) AS week_start,
    SUM(total_demand) AS total_demand
FROM {{ ref('daily_demand') }}
GROUP BY
    week_start
ORDER BY
    week_start