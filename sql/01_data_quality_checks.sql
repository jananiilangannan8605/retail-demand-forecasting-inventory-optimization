-- Week 1: BigQuery Data Quality Checks

-- 1. Calendar row count
SELECT COUNT(*) AS calendar_rows
FROM `blissful-axiom-509914-p6.retail_forecasting.calendar`;

-- 2. Sales validation row count
SELECT COUNT(*) AS validation_rows
FROM `blissful-axiom-509914-p6.retail_forecasting.sales_train_validation`;

-- 3. Sales evaluation row count
SELECT COUNT(*) AS evaluation_rows
FROM `blissful-axiom-509914-p6.retail_forecasting.sales_train_evaluation`;

-- 4. Sell prices row count
SELECT COUNT(*) AS sell_price_rows
FROM `blissful-axiom-509914-p6.retail_forecasting.sell_prices`;

-- 5. Calendar date range
SELECT
    MIN(date) AS start_date,
    MAX(date) AS end_date
FROM `blissful-axiom-509914-p6.retail_forecasting.calendar`;

-- 6. Check duplicate calendar dates
SELECT
    date,
    COUNT(*) AS row_count
FROM `blissful-axiom-509914-p6.retail_forecasting.calendar`
GROUP BY date
HAVING COUNT(*) > 1;