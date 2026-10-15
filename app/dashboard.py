import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="Retail Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# COLORFUL DARK THEME + SIDEBAR FIX
# ==========================================================
st.markdown("""
<style>
/* Main application background */
.stApp {
    background: linear-gradient(135deg, #0B1020, #111A30);
    color: #F8FAFC;
}

.block-container {
    max-width: 1600px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

[data-testid="stHeader"] {
    background: rgba(11, 16, 32, 0.9);
}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #151C30, #1C2540);
    border-right: 1px solid #303B59;
}

/* Sidebar headings and normal text */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #F1F5F9 !important;
}

/* Selectbox: light background and dark selected value */
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #F8FAFC !important;
    border: 1px solid #64748B !important;
    border-radius: 9px !important;
}

/* Selected value inside dropdown */
section[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #111827 !important;
}

/* Dropdown search/input text */
section[data-testid="stSidebar"] [data-baseweb="select"] input {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    caret-color: #111827 !important;
}

/* Dropdown arrow */
section[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #334155 !important;
    color: #334155 !important;
}

/* Dropdown popup background */
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] ul,
[role="listbox"] {
    background-color: #FFFFFF !important;
    color: #111827 !important;
    border: 1px solid #CBD5E1 !important;
}

/* Dropdown option text */
[role="option"] {
    background-color: #FFFFFF !important;
    color: #111827 !important;
}

/* Ensure nested dropdown text is readable */
[role="option"] *,
[role="listbox"] * {
    color: #111827 !important;
}

/* Dropdown hover effect */
[role="option"]:hover {
    background-color: #DBEAFE !important;
    color: #111827 !important;
}

/* Selected or highlighted dropdown option */
[role="option"][aria-selected="true"],
[role="option"][aria-selected="true"] * {
    background-color: #DBEAFE !important;
    color: #111827 !important;
}

/* Sidebar date input and number inputs */
section[data-testid="stSidebar"] input {
    background-color: #F8FAFC !important;
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    border-radius: 8px !important;
}

/* Sidebar date-picker calendar */
[data-baseweb="calendar"] {
    background-color: #FFFFFF !important;
    color: #111827 !important;
}

[data-baseweb="calendar"] *,
[data-baseweb="calendar"] button {
    color: #111827 !important;
}

/* Main dashboard headings */
.main-title {
    color: #FFFFFF !important;
    font-size: 2.2rem;
    font-weight: 800;
}

.subtitle {
    color: #AAB8D3 !important;
    font-size: 1rem;
}

.section-title {
    color: #FFFFFF !important;
    font-size: 1.35rem;
    font-weight: 750;
    border-left: 5px solid #06D6A0;
    padding-left: 12px;
    margin-top: 1.2rem;
    margin-bottom: 0.8rem;
}

/* Custom KPI cards */
.metric-card {
    background: linear-gradient(145deg, #19243B, #22213F) !important;
    border: 1px solid #34415F;
    border-top: 4px solid #06D6A0;
    border-radius: 15px;
    padding: 17px;
    min-height: 115px;
    box-shadow: 0 6px 22px rgba(0,0,0,0.18);
    transition: transform 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
}

.metric-title {
    color: #B8C5DF !important;
    font-size: 0.85rem;
    font-weight: 600;
}

.metric-value {
    color: #FFFFFF !important;
    font-size: 1.65rem;
    font-weight: 800;
    margin-top: 8px;
}

/* Streamlit metric cards */
[data-testid="stMetric"] {
    background: #19243B;
    border: 1px solid #34415F;
    border-radius: 12px;
    padding: 13px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}

[data-testid="stMetricDelta"] {
    color: #6EE7B7 !important;
}

[data-testid="stMarkdownContainer"] p {
    color: #E2E8F0;
}

/* Chart containers */
.stPlotlyChart {
    background: #151C30;
    border: 1px solid #293651;
    border-radius: 14px;
    padding: 7px;
}

/* Buttons */
.stDownloadButton button,
.stButton button {
    background: linear-gradient(90deg, #6366F1, #0891B2);
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    transition: all 0.2s ease;
}

.stDownloadButton button:hover,
.stButton button:hover {
    filter: brightness(1.15);
    transform: translateY(-1px);
}

hr {
    border-color: #303B59;
}

/* Slider accent */
.stSlider [data-baseweb="slider"] div {
    border-color: #06D6A0;
}

/* Responsive spacing */
@media (max-width: 768px) {
    .main-title {
        font-size: 1.7rem;
    }

    .metric-value {
        font-size: 1.3rem;
    }
}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# HEADER
# ==========================================================
st.markdown(
    '<div class="main-title">📊 Retail Intelligence Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Retail Demand Forecasting & Inventory '
    'Optimization | Prophet • BigQuery • dbt • Streamlit</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div style="display:flex; flex-wrap:wrap; gap:9px; margin:15px 0 20px 0;">
  <span style="background:#183B4E;color:#67E8F9;padding:7px 12px;border-radius:20px;">
  📈 Demand Forecasting</span>
  <span style="background:#153A32;color:#6EE7B7;padding:7px 12px;border-radius:20px;">
  📦 Inventory Planning</span>
  <span style="background:#342650;color:#D8B4FE;padding:7px 12px;border-radius:20px;">
  💡 What-if Analysis</span>
  <span style="background:#46301C;color:#FDBA74;padding:7px 12px;border-radius:20px;">
  🎯 Forecast Evaluation</span>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# LOAD DATA
# ==========================================================
project_root = Path(__file__).resolve().parents[1]

forecast_file = (
    project_root / "outputs" / "prophet" / "prophet_predictions.csv"
)

if not forecast_file.exists():
    st.error(f"Forecast CSV not found: {forecast_file}")
    st.info(
        "Confirm that outputs/prophet/prophet_predictions.csv "
        "exists in your project."
    )
    st.stop()

try:
    forecast_df = pd.read_csv(forecast_file)
except Exception as exc:
    st.error(f"Unable to read forecast file: {exc}")
    st.stop()

if forecast_df.empty:
    st.error("The forecast file contains no rows.")
    st.stop()


# ==========================================================
# DETECT COLUMNS
# ==========================================================
date_candidates = ["ds", "date", "Date", "forecast_date"]

prediction_candidates = [
    "yhat", "forecast", "prediction", "predicted_sales"
]

actual_candidates = [
    "actual_demand", "actual", "actual_sales", "y"
]

date_col = next(
    (c for c in date_candidates if c in forecast_df.columns), None
)

prediction_col = next(
    (c for c in prediction_candidates if c in forecast_df.columns), None
)

actual_col = next(
    (c for c in actual_candidates if c in forecast_df.columns), None
)

if date_col is None or prediction_col is None:
    st.error("Required date or forecast columns are missing.")
    st.write("Available columns:", list(forecast_df.columns))
    st.stop()

forecast_df[date_col] = pd.to_datetime(
    forecast_df[date_col], errors="coerce"
)

forecast_df[prediction_col] = pd.to_numeric(
    forecast_df[prediction_col], errors="coerce"
)

if actual_col is not None:
    forecast_df[actual_col] = pd.to_numeric(
        forecast_df[actual_col], errors="coerce"
    )

for bound_col in ["yhat_lower", "yhat_upper"]:
    if bound_col in forecast_df.columns:
        forecast_df[bound_col] = pd.to_numeric(
            forecast_df[bound_col], errors="coerce"
        )

forecast_df = forecast_df.dropna(
    subset=[date_col, prediction_col]
).copy()

if forecast_df.empty:
    st.error("No valid date and forecast values were found.")
    st.stop()

# Keep forecasts non-negative for demand and inventory planning.
forecast_df[prediction_col] = forecast_df[prediction_col].clip(lower=0)


# ==========================================================
# SIDEBAR FILTERS
# ==========================================================
st.sidebar.title("🎛️ Dashboard Controls")
st.sidebar.caption("Customize the data shown in the dashboard.")
st.sidebar.divider()

if "store_id" in forecast_df.columns:
    stores = sorted(
        forecast_df["store_id"].dropna().astype(str).unique()
    )

    store_options = ["All Stores"] + stores

    selected_store = st.sidebar.selectbox(
        "🏬 Select Store",
        store_options,
        key="store_filter"
    )
else:
    selected_store = "All Stores"

if "item_id" in forecast_df.columns:
    items = sorted(
        forecast_df["item_id"].dropna().astype(str).unique()
    )

    item_options = ["All Items"] + items

    selected_item = st.sidebar.selectbox(
        "📦 Select Item",
        item_options,
        key="item_filter"
    )
else:
    selected_item = "All Items"

min_date = forecast_df[date_col].min().date()
max_date = forecast_df[date_col].max().date()

selected_dates = st.sidebar.date_input(
    "🗓️ Forecast date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    help="Choose the forecast dates to analyse.",
    key="forecast_date_filter"
)

st.sidebar.divider()

st.sidebar.caption(
    "Forecast source: local Prophet prediction CSV"
)

st.sidebar.caption(
    f"Available forecast rows: {len(forecast_df):,}"
)


# ==========================================================
# APPLY FILTERS
# ==========================================================
filtered_df = forecast_df.copy()

if selected_store != "All Stores" and "store_id" in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df["store_id"].astype(str) == selected_store
    ]

if selected_item != "All Items" and "item_id" in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df["item_id"].astype(str) == selected_item
    ]

if isinstance(selected_dates, (tuple, list)):
    if len(selected_dates) == 2:
        start_date = pd.Timestamp(selected_dates[0])
        end_date = (
            pd.Timestamp(selected_dates[1]) + pd.Timedelta(days=1)
        )

        filtered_df = filtered_df[
            (filtered_df[date_col] >= start_date)
            & (filtered_df[date_col] < end_date)
        ]

    elif len(selected_dates) == 1:
        chosen_date = pd.Timestamp(selected_dates[0])

        filtered_df = filtered_df[
            filtered_df[date_col].dt.date == chosen_date.date()
        ]

elif selected_dates:
    chosen_date = pd.Timestamp(selected_dates)

    filtered_df = filtered_df[
        filtered_df[date_col].dt.date == chosen_date.date()
    ]

if filtered_df.empty:
    st.warning(
        "No rows match these filters. Please select another option."
    )
    st.stop()

filtered_df = filtered_df.sort_values(date_col).copy()


# ==========================================================
# KPI METRICS
# ==========================================================
total_forecast = float(filtered_df[prediction_col].sum())

average_daily_demand = float(
    filtered_df[prediction_col].mean()
)

peak_daily_demand = float(
    filtered_df[prediction_col].max()
)

forecast_days = int(
    filtered_df[date_col].nunique()
)

st.markdown(
    '<div class="section-title">✨ Forecast Overview</div>',
    unsafe_allow_html=True
)

k1, k2, k3, k4 = st.columns(4)

kpis = [
    (k1, "📅 Forecast Days", f"{forecast_days:,}", "#6366F1"),
    (k2, "📦 Total Forecast Demand", f"{total_forecast:,.0f}", "#06D6A0"),
    (k3, "📊 Average Daily Demand", f"{average_daily_demand:,.1f}", "#38BDF8"),
    (k4, "🔥 Peak Daily Demand", f"{peak_daily_demand:,.1f}", "#FB923C"),
]

for col, label, value, accent in kpis:
    with col:
        st.markdown(
            f"""
            <div class="metric-card" style="border-top-color:{accent};">
                <div class="metric-title">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================================
# FORECAST PERFORMANCE
# ==========================================================
st.markdown(
    '<div class="section-title">🎯 Actual vs Forecast Performance</div>',
    unsafe_allow_html=True
)

if actual_col is not None:
    comparison_df = filtered_df.dropna(
        subset=[actual_col, prediction_col]
    ).sort_values(date_col).copy()

    if not comparison_df.empty:
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

        e1, e2, e3 = st.columns(3)

        e1.metric("Mean Absolute Error (MAE)", f"{mae:,.2f}")
        e2.metric("Root Mean Squared Error (RMSE)", f"{rmse:,.2f}")
        e3.metric("Evaluation Rows", f"{len(comparison_df):,}")

        fig = go.Figure()

        # Prophet uncertainty interval, if present.
        if (
            "yhat_lower" in comparison_df.columns
            and "yhat_upper" in comparison_df.columns
        ):
            band = comparison_df.dropna(
                subset=["yhat_lower", "yhat_upper"]
            )

            if not band.empty:
                fig.add_trace(go.Scatter(
                    x=band[date_col],
                    y=band["yhat_upper"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    name="Upper interval"
                ))

                fig.add_trace(go.Scatter(
                    x=band[date_col],
                    y=band["yhat_lower"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(251,146,60,0.18)",
                    name="Prophet uncertainty interval",
                    hoverinfo="skip"
                ))

        fig.add_trace(go.Scatter(
            x=comparison_df[date_col],
            y=comparison_df[actual_col],
            mode="lines+markers",
            name="Actual Demand",
            line=dict(color="#06D6A0", width=3),
            marker=dict(size=6)
        ))

        fig.add_trace(go.Scatter(
            x=comparison_df[date_col],
            y=comparison_df[prediction_col],
            mode="lines+markers",
            name="Prophet Forecast",
            line=dict(color="#60A5FA", width=3, dash="dash"),
            marker=dict(size=5)
        ))

        fig.update_layout(
            title="Daily Actual Demand vs Prophet Forecast",
            xaxis_title="Date",
            yaxis_title="Demand (units)",
            template="plotly_dark",
            plot_bgcolor="#151C30",
            paper_bgcolor="#151C30",
            font=dict(color="#F8FAFC"),
            hovermode="x unified",
            height=440,
            margin=dict(l=20, r=20, t=65, b=20),
            legend=dict(orientation="h", y=1.12)
        )

        fig.update_xaxes(
            showgrid=True,
            gridcolor="#2A3650"
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor="#2A3650"
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("How to interpret MAE and RMSE"):
            st.write(
                "MAE is the average absolute difference between actual "
                "and predicted demand. RMSE gives greater weight to "
                "larger errors. Lower values generally indicate smaller "
                "forecast errors."
            )

    else:
        st.info(
            "Actual demand values are unavailable for the selected "
            "filters, so forecast error metrics cannot be calculated."
        )

else:
    st.info(
        "This CSV has no recognized actual-demand column. "
        "Actual-vs-forecast error metrics are unavailable."
    )


# ==========================================================
# DAILY COMPARISON BAR CHART
# ==========================================================
if actual_col is not None:
    st.markdown(
        '<div class="section-title">📊 Daily Demand Comparison</div>',
        unsafe_allow_html=True
    )

    bar_data = filtered_df.dropna(
        subset=[actual_col, prediction_col]
    ).sort_values(date_col)

    if not bar_data.empty:
        bar_fig = go.Figure()

        bar_fig.add_trace(go.Bar(
            x=bar_data[date_col],
            y=bar_data[actual_col],
            name="Actual",
            marker_color="#06D6A0"
        ))

        bar_fig.add_trace(go.Bar(
            x=bar_data[date_col],
            y=bar_data[prediction_col],
            name="Forecast",
            marker_color="#6366F1"
        ))

        bar_fig.update_layout(
            title="Actual and Forecast by Date",
            barmode="group",
            xaxis_title="Date",
            yaxis_title="Demand (units)",
            template="plotly_dark",
            plot_bgcolor="#151C30",
            paper_bgcolor="#151C30",
            font=dict(color="#F8FAFC"),
            hovermode="x unified",
            height=390,
            margin=dict(l=20, r=20, t=60, b=20),
            legend=dict(orientation="h", y=1.12)
        )

        bar_fig.update_yaxes(gridcolor="#2A3650")

        st.plotly_chart(
            bar_fig,
            use_container_width=True
        )


# ==========================================================
# STORE DEMAND DISTRIBUTION
# Uses selected dates, comparing all stores and items.
# ==========================================================
if "store_id" in forecast_df.columns:
    st.markdown(
        '<div class="section-title">🏬 Store Demand Distribution</div>',
        unsafe_allow_html=True
    )

    store_df = forecast_df.copy()

    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        s_date = pd.Timestamp(selected_dates[0])

        e_date = (
            pd.Timestamp(selected_dates[1]) + pd.Timedelta(days=1)
        )

        store_df = store_df[
            (store_df[date_col] >= s_date)
            & (store_df[date_col] < e_date)
        ]

    store_demand = (
        store_df.groupby("store_id", dropna=True)[prediction_col]
        .sum()
        .reset_index()
        .sort_values(prediction_col, ascending=False)
    )

    if not store_demand.empty:
        c1, c2 = st.columns(2)

        with c1:
            pie_fig = px.pie(
                store_demand,
                names="store_id",
                values=prediction_col,
                hole=0.55,
                title="Forecast Demand Share by Store",
                color_discrete_sequence=[
                    "#6366F1",
                    "#06D6A0",
                    "#F97316",
                    "#38BDF8",
                    "#C084FC",
                    "#F43F5E"
                ]
            )

            pie_fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate=(
                    "%{label}<br>Demand: %{value:,.1f}<br>"
                    "Share: %{percent}<extra></extra>"
                )
            )

            pie_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#151C30",
                plot_bgcolor="#151C30",
                font=dict(color="#F8FAFC"),
                height=390,
                margin=dict(l=20, r=20, t=60, b=20)
            )

            st.plotly_chart(
                pie_fig,
                use_container_width=True
            )

        with c2:
            store_bar = px.bar(
                store_demand,
                x="store_id",
                y=prediction_col,
                title="Total Forecast Demand by Store",
                text_auto=".0f",
                color="store_id",
                color_discrete_sequence=[
                    "#06D6A0",
                    "#6366F1",
                    "#F97316",
                    "#38BDF8",
                    "#C084FC",
                    "#F43F5E"
                ]
            )

            store_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="#151C30",
                plot_bgcolor="#151C30",
                font=dict(color="#F8FAFC"),
                showlegend=False,
                xaxis_title="Store",
                yaxis_title="Demand (units)",
                height=390,
                margin=dict(l=20, r=20, t=60, b=20)
            )

            store_bar.update_yaxes(
                gridcolor="#2A3650"
            )

            st.plotly_chart(
                store_bar,
                use_container_width=True
            )


# ==========================================================
# INVENTORY PLANNING
# ==========================================================
st.markdown(
    '<div class="section-title">📦 Inventory Planning</div>',
    unsafe_allow_html=True
)

inv1, inv2 = st.columns(2)

with inv1:
    current_inventory = st.number_input(
        "Current inventory (units)",
        min_value=0,
        value=0,
        step=10,
        help="Enter the stock currently available for the selected data."
    )

with inv2:
    lead_time_days = st.number_input(
        "Supplier lead time (days)",
        min_value=1,
        max_value=60,
        value=7,
        step=1
    )

safety_days = 3

safety_stock = average_daily_demand * safety_days

reorder_point = (
    average_daily_demand * lead_time_days
    + safety_stock
)

target_stock = (
    average_daily_demand * (lead_time_days + 7)
    + safety_stock
)

suggested_reorder_quantity = max(
    int(np.ceil(target_stock - current_inventory)),
    0
)

i1, i2, i3 = st.columns(3)

i1.metric(
    "🛡️ Safety Stock",
    f"{safety_stock:,.0f} units"
)

i2.metric(
    "🔔 Reorder Point",
    f"{reorder_point:,.0f} units"
)

i3.metric(
    "🛒 Suggested Reorder",
    f"{suggested_reorder_quantity:,} units"
)

if current_inventory <= reorder_point:
    st.warning(
        "Stock is at or below the estimated reorder point. "
        "Review the suggested replenishment quantity."
    )
else:
    st.success(
        "Current stock is above the estimated reorder point."
    )

st.caption(
    "Planning assumptions: safety stock = 3 days of average forecast "
    "demand; target stock = supplier lead time + 7 additional days + "
    "safety stock. This is a simplified estimate, not a complete "
    "inventory simulation."
)


# ==========================================================
# PRICE-DROP WHAT-IF ANALYSIS
# ==========================================================
st.markdown(
    '<div class="section-title">💡 Price-Drop What-if Analysis</div>',
    unsafe_allow_html=True
)

price_drop = st.slider(
    "Simulated price reduction (%)",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

# Illustrative assumption only; not learned from the dataset.
demand_lift_rate = 1.5

estimated_demand_increase = price_drop * demand_lift_rate

adjusted_demand = total_forecast * (
    1 + estimated_demand_increase / 100
)

additional_demand = adjusted_demand - total_forecast

p1, p2, p3 = st.columns(3)

p1.metric(
    "Price Reduction",
    f"{price_drop}%"
)

p2.metric(
    "Assumed Demand Increase",
    f"{estimated_demand_increase:.1f}%"
)

p3.metric(
    "Adjusted Forecast Demand",
    f"{adjusted_demand:,.0f}",
    delta=f"+{additional_demand:,.0f}"
)

scenario_df = pd.DataFrame({
    "Scenario": [
        "Current Forecast",
        "Price-Drop Scenario"
    ],
    "Demand": [
        total_forecast,
        adjusted_demand
    ]
})

scenario_fig = px.bar(
    scenario_df,
    x="Scenario",
    y="Demand",
    color="Scenario",
    text_auto=".0f",
    title="Current Forecast vs Simulated Scenario",
    color_discrete_sequence=[
        "#38BDF8",
        "#C084FC"
    ]
)

scenario_fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#151C30",
    plot_bgcolor="#151C30",
    font=dict(color="#F8FAFC"),
    showlegend=False,
    yaxis_title="Demand (units)",
    height=360,
    margin=dict(l=20, r=20, t=60, b=20)
)

scenario_fig.update_yaxes(
    gridcolor="#2A3650"
)

st.plotly_chart(
    scenario_fig,
    use_container_width=True
)

st.caption(
    "Demonstration assumption: every 1% price reduction corresponds "
    "to a 1.5% demand increase. This is not a causal estimate or a "
    "model trained to predict price elasticity."
)


# ==========================================================
# FORECAST DETAILS AND CSV DOWNLOAD
# ==========================================================
st.markdown(
    '<div class="section-title">📋 Forecast Details</div>',
    unsafe_allow_html=True
)

display_columns = [
    c for c in [
        date_col,
        "store_id",
        "item_id",
        actual_col,
        prediction_col,
        "yhat_lower",
        "yhat_upper"
    ]
    if c is not None and c in filtered_df.columns
]

display_df = filtered_df[display_columns].copy()

rename_map = {
    date_col: "Date",
    "store_id": "Store",
    "item_id": "Item",
    prediction_col: "Forecast Demand",
    "yhat_lower": "Forecast Lower Bound",
    "yhat_upper": "Forecast Upper Bound"
}

if actual_col is not None:
    rename_map[actual_col] = "Actual Demand"

display_df = display_df.rename(
    columns=rename_map
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download Filtered Forecast CSV",
    data=csv_data,
    file_name="retail_demand_forecast.csv",
    mime="text/csv",
    use_container_width=True
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
    "Python • Pandas • Prophet • BigQuery • dbt • Streamlit • Plotly"
)
