* [**README**](https://github.com/jananiilangannan8605/retail-demand-forecasting-inventory-optimization#)

# 📈 Retail Demand Forecasting & Inventory Optimization

A data analytics and machine learning system designed to forecast retail product demand and support inventory planning. The project uses the M5 Walmart retail dataset, Google BigQuery, dbt, Prophet, and LightGBM to transform historical sales data into demand forecasts and inventory insights.

## 📌 Project Overview

| **Feature**                | **Description**                               |
| -------------------------- | --------------------------------------------- |
| **Data Ingestion**         | Load retail sales data into Google BigQuery   |
| **Data Quality**           | SQL-based schema and missing-value validation |
| **Data Transformation**    | dbt models for clean and structured datasets  |
| **Demand Forecasting**     | Prophet and LightGBM forecasting approaches   |
| **Feature Engineering**    | Lag and rolling demand features               |
| **Inventory Optimization** | Support demand-based inventory planning       |
| **Dashboard**              | Streamlit-based interactive visualization     |

## 🛠️ Tech Stack

* **Language:** Python, SQL
* **Data Warehouse:** Google BigQuery
* **Transformation:** dbt
* **ML & Forecasting:** Prophet, LightGBM, Scikit-Learn
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn, Streamlit
* **Version Control:** Git + GitHub

## 📊 Dataset

* **Source:** M5 Walmart Retail Sales Dataset
* **Dataset Type:** Retail sales forecasting
* **Main Files:** `calendar.csv`, `sales_train_validation.csv`, `sales_train_evaluation.csv`, `sell_prices.csv`
* **Data Includes:** Store, item, department, state, daily sales, prices, calendar events and SNAP information

## 📁 Project Structure

```text
retail-demand-forecasting-inventory-optimization/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── featured_train.csv
├── sql/
│   ├── 01_data_quality_checks.sql
│   └── 02_schema_and_missing_value_checks.sql
├── src/
│   ├── preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_analysis.py
│   ├── inventory_optimization.py
│   ├── bigquery_client.py
│   ├── data_loader.py
│   └── prophet_forecasting.py
├── dbt/
│   └── models/
├── models/
├── outputs/
└── app/
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/jananiilangannan8605/retail-demand-forecasting-inventory-optimization.git
cd retail-demand-forecasting-inventory-optimization
```

### 2. Setup Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Google Cloud

```bash
gcloud auth login
gcloud config set project blissful-axiom-509914-p6
```

### 4. Run Data Pipeline

Load the M5 dataset into BigQuery and run the SQL data-quality checks.

The project uses BigQuery dataset:

```text
retail_forecasting
```

### 5. Run Forecasting

```bash
python src/model_training.py
```

Prophet forecasting can be run using:

```bash
python src/prophet_forecasting.py
```

## 📅 Development Timeline

| **Week** | **Focus Area**                            |
| -------- | ----------------------------------------- |
| Week 1   | Data Ingestion, BigQuery & Data Quality   |
| Week 2   | dbt Transformation & Forecast Preparation |
| Week 3   | Prophet & LightGBM Forecasting            |
| Week 4   | Streamlit Dashboard & Reporting           |

## 📌 Current Progress

* ✅ M5 dataset preprocessing
* ✅ BigQuery data ingestion
* ✅ Data quality checks
* ✅ dbt transformation models
* ✅ Daily, weekly and monthly demand datasets
* ✅ Prophet forecasting baseline
* ✅ LightGBM demand forecasting
* 🚧 Streamlit dashboard and what-if analysis

## 👨‍💻 Author


**Zaalima Development Internship Project**

**Project:** Retail Demand Forecasting & Inventory Optimization
## 📄 License

This project is developed as part of the **Zaalima Development Internship** and is intended for educational, internship, and portfolio purposes.


