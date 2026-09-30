{{ config(materialized='view') }}

WITH date_boundary AS (

    SELECT
        MAX(date) AS max_date
    FROM {{ ref('forecast_base') }}

)

SELECT
    f.date,
    f.d,
    f.store_id,
    f.item_id,
    f.dept_id,
    f.cat_id,
    f.state_id,
    f.demand,

    CASE
        WHEN f.date < DATE_SUB(b.max_date, INTERVAL 29 DAY)
        THEN 'train'
        ELSE 'validation'
    END AS dataset_type

FROM {{ ref('forecast_base') }} f
CROSS JOIN date_boundary b