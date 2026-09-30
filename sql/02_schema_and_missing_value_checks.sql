-- Week 1: Schema and Missing-Value Checks
-- Project: Retail Demand Forecasting & Inventory Optimization


-- ============================================================
-- 1. CALENDAR - Missing values
-- ============================================================

SELECT
    COUNT(*) AS total_rows,
    COUNTIF(date IS NULL) AS missing_date,
    COUNTIF(wm_yr_wk IS NULL) AS missing_wm_yr_wk,
    COUNTIF(weekday IS NULL) AS missing_weekday,
    COUNTIF(wday IS NULL) AS missing_wday,
    COUNTIF(month IS NULL) AS missing_month,
    COUNTIF(year IS NULL) AS missing_year,
    COUNTIF(d IS NULL) AS missing_d
FROM `blissful-axiom-509914-p6.retail_forecasting.calendar`;


-- ============================================================
-- 2. SELL PRICES - Missing values
-- ============================================================

SELECT
    COUNT(*) AS total_rows,
    COUNTIF(store_id IS NULL) AS missing_store_id,
    COUNTIF(item_id IS NULL) AS missing_item_id,
    COUNTIF(wm_yr_wk IS NULL) AS missing_wm_yr_wk,
    COUNTIF(sell_price IS NULL) AS missing_sell_price
FROM `blissful-axiom-509914-p6.retail_forecasting.sell_prices`;


-- ============================================================
-- 3. SALES VALIDATION - Missing values
-- ============================================================

SELECT
    COUNT(*) AS total_rows,
    COUNTIF(id IS NULL) AS missing_id,
    COUNTIF(item_id IS NULL) AS missing_item_id,
    COUNTIF(dept_id IS NULL) AS missing_dept_id,
    COUNTIF(cat_id IS NULL) AS missing_cat_id,
    COUNTIF(store_id IS NULL) AS missing_store_id,
    COUNTIF(state_id IS NULL) AS missing_state_id
FROM `blissful-axiom-509914-p6.retail_forecasting.sales_train_validation`;


-- ============================================================
-- 4. SALES EVALUATION - Missing values
-- ============================================================

SELECT
    COUNT(*) AS total_rows,
    COUNTIF(id IS NULL) AS missing_id,
    COUNTIF(item_id IS NULL) AS missing_item_id,
    COUNTIF(dept_id IS NULL) AS missing_dept_id,
    COUNTIF(cat_id IS NULL) AS missing_cat_id,
    COUNTIF(store_id IS NULL) AS missing_store_id,
    COUNTIF(state_id IS NULL) AS missing_state_id
FROM `blissful-axiom-509914-p6.retail_forecasting.sales_train_evaluation`;


-- ============================================================
-- 5. Calendar date range
-- ============================================================

SELECT
    MIN(date) AS earliest_date,
    MAX(date) AS latest_date,
    DATE_DIFF(MAX(date), MIN(date), DAY) + 1 AS calendar_days
FROM `blissful-axiom-509914-p6.retail_forecasting.calendar`;


-- ============================================================
-- 6. Sales dimensions
-- ============================================================

SELECT
    COUNT(DISTINCT item_id) AS unique_items,
    COUNT(DISTINCT dept_id) AS unique_departments,
    COUNT(DISTINCT cat_id) AS unique_categories,
    COUNT(DISTINCT store_id) AS unique_stores,
    COUNT(DISTINCT state_id) AS unique_states
FROM `blissful-axiom-509914-p6.retail_forecasting.sales_train_validation`;