{{ config(materialized='view') }}

{% set sales_columns = get_sales_columns() %}

WITH sales_data AS (

    SELECT *
    FROM `blissful-axiom-509914-p6.retail_forecasting.sales_train_validation`

),

sales_long AS (

    SELECT
        id,
        item_id,
        dept_id,
        cat_id,
        store_id,
        state_id,
        d,
        demand
    FROM sales_data
    UNPIVOT (
        demand FOR d IN (
            {% for column in sales_columns %}
                {{ column }}{% if not loop.last %}, {% endif %}
            {% endfor %}
        )
    )

)

SELECT
    s.id,
    s.item_id,
    s.dept_id,
    s.cat_id,
    s.store_id,
    s.state_id,
    c.date,
    s.d,
    s.demand
FROM sales_long s
JOIN `blissful-axiom-509914-p6.retail_forecasting.calendar` c
    ON c.d = s.d