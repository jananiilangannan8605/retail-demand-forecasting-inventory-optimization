import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background-color: #F8FAFC;
        color: #172033;
    }

    /* Main content text */
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #172033 !important;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #475467 !important;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Section headings */
    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        color: #172033 !important;
        margin-top: 1rem;
        margin-bottom: 0.7rem;
    }

    /* KPI cards */
    .metric-card {
        background-color: #FFFFFF !important;
        padding: 1.1rem;
        border-radius: 14px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.06);
        min-height: 115px;
    }

    .metric-title {
        color: #475467 !important;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .metric-value {
        color: #172033 !important;
        font-size: 1.65rem;
        font-weight: 700;
        margin-top: 0.35rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
    }

    section[data-testid="stSidebar"] * {
        color: #172033 !important;
    }

    /* Sidebar select boxes */
    section[data-testid="stSidebar"] label {
        color: #172033 !important;
        font-weight: 600;
    }

    /* Normal text */
    .stMarkdown,
    .stText,
    p,
    span,
    label {
        color: #172033;
    }

    /* Metric component */
    [data-testid="stMetricLabel"] {
        color: #475467 !important;
    }

    [data-testid="stMetricValue"] {
        color: #172033 !important;
    }

    [data-testid="stMetricDelta"] {
        color: #16A34A !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        background-color: #FFFFFF !important;
    }

    /* Info box */
    [data-testid="stAlert"] {
        color: #172033 !important;
    }

    /* Slider text */
    [data-testid="stSlider"] label {
        color: #172033 !important;
    }

    /* Download button */
    .stDownloadButton button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }

    .stDownloadButton button:hover {
        background-color: #1D4ED8 !important;
        color: #FFFFFF !important;
    }

    /* Divider */
    hr {
        border-color: #E2E8F0 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)
#==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">📊 Retail Demand Forecasting & Inventory Optimization</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Interactive analytics dashboard for demand forecasting, inventory planning, '
    'forecast evaluation, and price-based what-if analysis.'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================================
# LOAD FORECAST DATA
# ==========================================================

project_root = Path(__file__).resolve().parents[1]

forecast_file = (
    project_root
    / "outputs"
    / "prophet"
    / "prophet_predictions.csv"
)

if not forecast_file.exists():
    st.error(f"Forecast file not found: {forecast_file}")
    st.stop()

forecast_df = pd.read_csv(forecast_file)


# ==========================================================
# DETECT COLUMNS
# ==========================================================

date_candidates = [
    "ds",
    "date",
    "Date",
    "forecast_date"
]

prediction_candidates = [
    "yhat",
    "forecast",
    "prediction",
    "predicted_sales"
]

actual_candidates = [
    "actual_demand",
    "actual",
    "actual_sales",
    "y"
]

date_col = next(
    (
        col
        for col in date_candidates
        if col in forecast_df.columns
    ),
    None
)

prediction_col = next(
    (
        col
        for col in prediction_candidates
        if col in forecast_df.columns
    ),
    None
)

actual_col = next(
    (
        col
        for col in actual_candidates
        if col in forecast_df.columns
    ),
    None
)

if date_col is None or prediction_col is None:
    st.error(
        "Required forecast columns were not found."
    )

    st.write(
        "Available columns:",
        list(forecast_df.columns)
    )

    st.stop()


forecast_df[date_col] = pd.to_datetime(
    forecast_df[date_col]
)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🎛️ Dashboard Controls")

st.sidebar.markdown(
    "Use the filters below to explore individual store and item forecasts."
)

st.sidebar.divider()


# Store filter
if "store_id" in forecast_df.columns:

    stores = sorted(
        forecast_df["store_id"]
        .astype(str)
        .unique()
    )

    selected_store = st.sidebar.selectbox(
        "🏬 Select Store",
        stores
    )

else:
    selected_store = None


# Item filter
if "item_id" in forecast_df.columns:

    items = sorted(
        forecast_df["item_id"]
        .astype(str)
        .unique()
    )

    selected_item = st.sidebar.selectbox(
        "📦 Select Item",
        items
    )

else:
    selected_item = None


st.sidebar.divider()

st.sidebar.caption(
    "Dashboard powered by Prophet demand forecasting "
    "and inventory planning analysis."
)


# ==========================================================
# APPLY FILTERS
# ==========================================================

filtered_df = forecast_df.copy()

if selected_store is not None:

    filtered_df = filtered_df[
        filtered_df["store_id"].astype(str)
        == selected_store
    ]


if selected_item is not None:

    filtered_df = filtered_df[
        filtered_df["item_id"].astype(str)
        == selected_item
    ]


if filtered_df.empty:

    st.warning(
        "No forecast data is available for the selected filters."
    )

    st.stop()


# ==========================================================
# BASIC METRICS
# ==========================================================

total_forecast = filtered_df[prediction_col].sum()

average_daily_demand = (
    filtered_df[prediction_col].mean()
)

maximum_daily_demand = (
    filtered_df[prediction_col].max()
)

forecast_days = (
    filtered_df[date_col].nunique()
)


# ==========================================================
# KPI CARDS
# ==========================================================

st.markdown(
    '<div class="section-title">📈 Forecast Overview</div>',
    unsafe_allow_html=True
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)


with kpi1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Forecast Days</div>
            <div class="metric-value">{forecast_days}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with kpi2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Total Forecast Demand</div>
            <div class="metric-value">{total_forecast:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with kpi3:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Average Daily Demand</div>
            <div class="metric-value">{average_daily_demand:,.1f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with kpi4:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Peak Daily Demand</div>
            <div class="metric-value">{maximum_daily_demand:,.1f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


# ==========================================================
# ACTUAL VS FORECAST
# ==========================================================

if actual_col is not None:

    st.markdown(
        '<div class="section-title">🎯 Actual vs Forecast Performance</div>',
        unsafe_allow_html=True
    )

    comparison_df = filtered_df[
        [
            date_col,
            prediction_col,
            actual_col
        ]
    ].copy()

    comparison_df = comparison_df.sort_values(
        date_col
    )

    comparison_df = comparison_df.dropna(
        subset=[
            prediction_col,
            actual_col
        ]
    )

    if not comparison_df.empty:

        # --------------------------------------------------
        # Accuracy metrics
        # --------------------------------------------------

        mae = mean_absolute_error(
            comparison_df[actual_col],
            comparison_df[prediction_col]
        )

        rmse = np.sqrt(
            mean_squared_error(
                comparison_df[actual_col],
                comparison_df[prediction_col]
            )
        )

        error_col1, error_col2 = st.columns(2)

        with error_col1:

            st.metric(
                "Mean Absolute Error (MAE)",
                f"{mae:,.2f}"
            )

        with error_col2:

            st.metric(
                "Root Mean Squared Error (RMSE)",
                f"{rmse:,.2f}"
            )


        # --------------------------------------------------
        # Actual vs Forecast Line Chart
        # --------------------------------------------------

        line_fig = go.Figure()

        line_fig.add_trace(
            go.Scatter(
                x=comparison_df[date_col],
                y=comparison_df[actual_col],
                mode="lines+markers",
                name="Actual Demand",
                line=dict(
                    color="#2563EB",
                    width=3
                ),
                marker=dict(
                    size=6
                )
            )
        )

        line_fig.add_trace(
            go.Scatter(
                x=comparison_df[date_col],
                y=comparison_df[prediction_col],
                mode="lines+markers",
                name="Forecast Demand",
                line=dict(
                    color="#F97316",
                    width=3,
                    dash="dash"
                ),
                marker=dict(
                    size=5
                )
            )
        )

        line_fig.update_layout(
            title="Actual Demand vs Prophet Forecast",
            xaxis_title="Date",
            yaxis_title="Demand",
            hovermode="x unified",
            height=450,
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(
                l=20,
                r=20,
                t=70,
                b=20
            )
        )

        line_fig.update_xaxes(
            showgrid=True,
            gridcolor="#E5E7EB"
        )

        line_fig.update_yaxes(
            showgrid=True,
            gridcolor="#E5E7EB"
        )

        st.plotly_chart(
            line_fig,
            width="stretch"
        )


        # --------------------------------------------------
        # Actual vs Forecast Bar Chart
        # --------------------------------------------------

        st.markdown(
            '<div class="section-title">📊 Daily Demand Comparison</div>',
            unsafe_allow_html=True
        )

        bar_fig = go.Figure()

        bar_fig.add_trace(
            go.Bar(
                x=comparison_df[date_col],
                y=comparison_df[actual_col],
                name="Actual Demand",
                marker_color="#2563EB"
            )
        )

        bar_fig.add_trace(
            go.Bar(
                x=comparison_df[date_col],
                y=comparison_df[prediction_col],
                name="Forecast Demand",
                marker_color="#F97316"
            )
        )

        bar_fig.update_layout(
            title="Daily Actual vs Forecast Demand",
            xaxis_title="Date",
            yaxis_title="Demand",
            barmode="group",
            height=430,
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(
                l=20,
                r=20,
                t=70,
                b=20
            )
        )

        bar_fig.update_xaxes(
            showgrid=False
        )

        bar_fig.update_yaxes(
            showgrid=True,
            gridcolor="#E5E7EB"
        )

        st.plotly_chart(
            bar_fig,
            width="stretch"
        )


# ==========================================================
# STORE DEMAND SHARE
# ==========================================================

if "store_id" in forecast_df.columns:

    st.markdown(
        '<div class="section-title">🏬 Demand Distribution</div>',
        unsafe_allow_html=True
    )

    store_demand = (
        forecast_df
        .groupby("store_id")[prediction_col]
        .sum()
        .reset_index()
    )

    store_demand.columns = [
        "Store",
        "Forecast Demand"
    ]

    pie_col, bar_col = st.columns(2)


    # ------------------------------------------------------
    # Pie Chart
    # ------------------------------------------------------

    with pie_col:

        pie_fig = px.pie(
            store_demand,
            names="Store",
            values="Forecast Demand",
            title="Forecast Demand Share by Store",
            hole=0.45
        )

        pie_fig.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        pie_fig.update_layout(
            height=430,
            paper_bgcolor="white",
            margin=dict(
                l=20,
                r=20,
                t=70,
                b=20
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            )
        )

        st.plotly_chart(
            pie_fig,
            width="stretch"
        )


    # ------------------------------------------------------
    # Store Bar Chart
    # ------------------------------------------------------

    with bar_col:

        store_bar_fig = px.bar(
            store_demand,
            x="Store",
            y="Forecast Demand",
            text_auto=".0f",
            title="Forecast Demand by Store"
        )

        store_bar_fig.update_traces(
            marker_color="#14B8A6"
        )

        store_bar_fig.update_layout(
            height=430,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis_title="Store",
            yaxis_title="Forecast Demand",
            margin=dict(
                l=20,
                r=20,
                t=70,
                b=20
            )
        )

        store_bar_fig.update_yaxes(
            showgrid=True,
            gridcolor="#E5E7EB"
        )

        st.plotly_chart(
            store_bar_fig,
            width="stretch"
        )


# ==========================================================
# INVENTORY PLANNING
# ==========================================================

st.markdown(
    '<div class="section-title">📦 Inventory Planning</div>',
    unsafe_allow_html=True
)


# Simple safety-stock calculation
safety_stock = (
    average_daily_demand * 3
)

reorder_point = (
    average_daily_demand * 7
    + safety_stock
)

suggested_reorder_quantity = max(
    total_forecast - safety_stock,
    0
)


inventory_col1, inventory_col2, inventory_col3 = st.columns(3)


with inventory_col1:

    st.metric(
        "Safety Stock",
        f"{safety_stock:,.0f}"
    )


with inventory_col2:

    st.metric(
        "Reorder Point",
        f"{reorder_point:,.0f}"
    )


with inventory_col3:

    st.metric(
        "Suggested Reorder Quantity",
        f"{suggested_reorder_quantity:,.0f}"
    )


st.info(
    "Inventory values are planning estimates based on forecast demand. "
    "Safety stock is calculated using approximately 3 days of average demand."
)


# ==========================================================
# PRICE-DROP WHAT-IF ANALYSIS
# ==========================================================

st.markdown(
    '<div class="section-title">💰 Price-Drop What-If Analysis</div>',
    unsafe_allow_html=True
)


price_drop = st.slider(
    "Simulated Price Reduction (%)",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)


# Demonstration elasticity assumption
demand_lift_rate = 1.5

estimated_demand_increase = (
    price_drop * demand_lift_rate
)

adjusted_demand = (
    total_forecast
    * (
        1
        + estimated_demand_increase / 100
    )
)

additional_demand = (
    adjusted_demand
    - total_forecast
)


price_col1, price_col2, price_col3 = st.columns(3)


with price_col1:

    st.metric(
        "Price Reduction",
        f"{price_drop}%"
    )


with price_col2:

    st.metric(
        "Estimated Demand Increase",
        f"{estimated_demand_increase:.1f}%"
    )


with price_col3:

    st.metric(
        "Adjusted Forecast Demand",
        f"{adjusted_demand:,.0f}",
        delta=f"+{additional_demand:,.0f}"
    )


# ----------------------------------------------------------
# What-if Bar Chart
# ----------------------------------------------------------

what_if_df = pd.DataFrame(
    {
        "Scenario": [
            "Current Forecast",
            "Price-Drop Scenario"
        ],
        "Demand": [
            total_forecast,
            adjusted_demand
        ]
    }
)

what_if_fig = px.bar(
    what_if_df,
    x="Scenario",
    y="Demand",
    text_auto=".0f",
    title="Current vs Price-Drop Demand Scenario"
)

what_if_fig.update_traces(
    marker_color=[
        "#64748B",
        "#8B5CF6"
    ]
)

what_if_fig.update_layout(
    height=380,
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis_title="Scenario",
    yaxis_title="Forecast Demand",
    margin=dict(
        l=20,
        r=20,
        t=70,
        b=20
    )
)

what_if_fig.update_yaxes(
    showgrid=True,
    gridcolor="#E5E7EB"
)

st.plotly_chart(
    what_if_fig,
    width="stretch"
)


st.caption(
    "What-if analysis uses a simple assumed demand elasticity of 1.5x "
    "for demonstration purposes. It is not a causal estimate."
)


# ==========================================================
# FORECAST DATA TABLE
# ==========================================================

st.markdown(
    '<div class="section-title">📋 Forecast Details</div>',
    unsafe_allow_html=True
)


display_columns = [
    col
    for col in [
        date_col,
        "store_id",
        "item_id",
        prediction_col,
        actual_col
    ]
    if col is not None
    and col in filtered_df.columns
]


display_df = filtered_df[
    display_columns
].copy()


# Rename columns for presentation
rename_map = {
    date_col: "Date",
    prediction_col: "Forecast Demand"
}

if actual_col is not None:
    rename_map[actual_col] = "Actual Demand"

if "store_id" in display_df.columns:
    rename_map["store_id"] = "Store"

if "item_id" in display_df.columns:
    rename_map["item_id"] = "Item"


display_df = display_df.rename(
    columns=rename_map
)


st.dataframe(
    display_df,
    width="stretch",
    hide_index=True
)


# ==========================================================
# DOWNLOAD
# ==========================================================

csv_data = filtered_df.to_csv(
    index=False
)

st.download_button(
    label="⬇️ Download Forecast CSV",
    data=csv_data,
    file_name="retail_demand_forecast.csv",
    mime="text/csv",
    width="stretch"
)


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Retail Demand Forecasting & Inventory Optimization | "
    "Zaalima Development Internship Project"
)

st.caption(
    "Technology: Python • BigQuery • dbt • Prophet • LightGBM • Streamlit • Plotly"
)