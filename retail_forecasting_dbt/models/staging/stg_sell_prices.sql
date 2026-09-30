{{ config(materialized='view') }}

SELECT
    store_id,
    item_id,
    wm_yr_wk,
    sell_price
FROM `blissful-axiom-509914-p6.retail_forecasting.sell_prices`