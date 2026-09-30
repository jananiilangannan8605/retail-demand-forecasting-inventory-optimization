{{ config(materialized='table') }}

SELECT
    date,
    d,
    store_id,
    item_id,
    dept_id,
    cat_id,
    state_id,
    demand
FROM {{ ref('stg_sales') }}
WHERE demand >= 0
ORDER BY
    store_id,
    item_id,
    date