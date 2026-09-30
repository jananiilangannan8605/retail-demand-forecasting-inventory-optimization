{{ config(materialized='view') }}

SELECT
    date AS ds,
    demand AS y,
    store_id,
    item_id
FROM {{ ref('forecast_split') }}
WHERE dataset_type = 'train'
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
ORDER BY
    store_id,
    item_id,
    ds