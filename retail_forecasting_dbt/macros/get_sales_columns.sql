{% macro get_sales_columns() %}

    {% set query %}
        SELECT column_name
        FROM `blissful-axiom-509914-p6.retail_forecasting.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = 'sales_train_validation'
          AND STARTS_WITH(column_name, 'd_')
        ORDER BY ordinal_position
    {% endset %}

    {% set results = run_query(query) %}

    {% if execute %}
        {% set sales_columns = [] %}
        {% for row in results.rows %}
            {% do sales_columns.append(row[0]) %}
        {% endfor %}
    {% else %}
        {% set sales_columns = [] %}
    {% endif %}

    {{ return(sales_columns) }}

{% endmacro %}