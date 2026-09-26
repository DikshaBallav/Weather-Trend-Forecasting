import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from statsmodels.tsa.holtwinters import ExponentialSmoothing


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Global Weather Trend & Forecasting",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 40px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 30px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #f5f7fa;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

# Public Hugging Face dataset URL.
# This is used as the default source when the app is deployed.
DEFAULT_DATA_URL = (
    "https://huggingface.co/datasets/dikshaballav/"
    "GlobalWeatherRepository/resolve/main/GlobalWeatherRepository.csv"
)


@st.cache_data(show_spinner="Loading weather dataset...")
def load_data():

    # ------------------------------------------------------------
    # 1. Streamlit Cloud / deployed app:
    #    If DATA_URL is configured in Manage App -> Settings -> Secrets,
    #    use that URL.
    # ------------------------------------------------------------
    data_url = None

    try:
        data_url = st.secrets.get("DATA_URL")
    except Exception:
        # No Streamlit secret configured.
        data_url = None

    if data_url:
        try:
            return pd.read_csv(data_url)
        except Exception as e:
            st.warning(
                "The DATA_URL configured in Streamlit Secrets could not "
                f"be loaded. Falling back to the public Hugging Face dataset. "
                f"Details: {e}"
            )

    # ------------------------------------------------------------
    # 2. Default deployed source:
    #    Public Hugging Face dataset.
    # ------------------------------------------------------------
    try:
        return pd.read_csv(DEFAULT_DATA_URL)
    except Exception as e:
        st.warning(
            "The Hugging Face dataset could not be loaded. "
            f"Details: {e}"
        )

    # ------------------------------------------------------------
    # 3. Local development fallback:
    #    This allows the app to continue working on your laptop
    #    when the CSV exists in the local data folder.
    # ------------------------------------------------------------
    possible_paths = [
        "data/GlobalWeatherRepository.csv",
        "GlobalWeatherRepository.csv"
    ]

    for path in possible_paths:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        "GlobalWeatherRepository.csv could not be loaded. "
        "For Streamlit Cloud, verify the Hugging Face dataset URL "
        "or configure DATA_URL in Manage App -> Settings -> Secrets. "
        "For local use, place the CSV inside the data folder."
    )


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_column(df, possible_names):

    lower_columns = {
        col.lower().strip(): col
        for col in df.columns
    }

    for name in possible_names:

        if name.lower() in lower_columns:
            return lower_columns[name.lower()]

    return None


def detect_columns(df):

    columns = {}

    columns["date"] = find_column(
        df,
        [
            "last_updated",
            "lastupdated",
            "last updated",
            "date",
            "datetime",
            "timestamp"
        ]
    )

    columns["country"] = find_column(
        df,
        [
            "country"
        ]
    )

    columns["location"] = find_column(
        df,
        [
            "location_name",
            "location",
            "city"
        ]
    )

    columns["temperature"] = find_column(
        df,
        [
            # Actual column name in GlobalWeatherRepository.csv
            "temperature_celsius",
            "temperature_c",
            "temp_celsius",
            "temp_c",
            "temperature",
            "temp"
        ]
    )

    columns["precipitation"] = find_column(
        df,
        [
            "precip_mm",
            "precipitation_mm",
            "precipitation",
            "precip_mm"
        ]
    )

    columns["humidity"] = find_column(
        df,
        [
            "humidity"
        ]
    )

    columns["wind"] = find_column(
        df,
        [
            "wind_kph",
            "wind_speed",
            "wind_speed_kph"
        ]
    )

    columns["latitude"] = find_column(
        df,
        [
            "latitude",
            "lat"
        ]
    )

    columns["longitude"] = find_column(
        df,
        [
            "longitude",
            "lon",
            "lng"
        ]
    )

    columns["pm25"] = find_column(
        df,
        [
            "air_quality_pm2.5",
            "air_quality_pm2_5",
            "pm2.5",
            "pm25"
        ]
    )

    columns["pm10"] = find_column(
        df,
        [
            "air_quality_pm10",
            "pm10"
        ]
    )

    return columns


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_data(df, columns):

    df = df.copy()

    date_col = columns["date"]

    if date_col:

        df[date_col] = pd.to_datetime(
            df[date_col],
            errors="coerce"
        )

        df = df.dropna(subset=[date_col])

        df = df.sort_values(date_col)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Convert numeric columns
    numeric_columns = df.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Fill numerical missing values
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_columns:

        if df[col].isna().sum() > 0:

            df[col] = df[col].fillna(
                df[col].median()
            )

    # Fill categorical missing values
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in categorical_columns:

        if df[col].isna().sum() > 0:

            mode = df[col].mode()

            if len(mode) > 0:
                df[col] = df[col].fillna(mode.iloc[0])

    return df


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    non_zero = actual != 0

    if np.sum(non_zero) > 0:

        mape = np.mean(
            np.abs(
                (actual[non_zero] -
                 predicted[non_zero])
                /
                actual[non_zero]
            )
        ) * 100

    else:

        mape = np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "R²": r2
    }


# ============================================================
# SEASONAL NAIVE FORECAST
# ============================================================

def seasonal_naive_forecast(
    train,
    horizon,
    season_length=7
):

    values = train.values

    predictions = []

    for i in range(horizon):

        index = len(values) - season_length + (
            i % season_length
        )

        predictions.append(
            values[index]
        )

    return np.array(predictions)


# ============================================================
# HOLT-WINTERS FORECAST
# ============================================================

def holt_winters_forecast(
    train,
    horizon
):

    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=7,
        initialization_method="estimated"
    )

    fitted_model = model.fit(
        optimized=True
    )

    forecast = fitted_model.forecast(
        horizon
    )

    return np.array(forecast)


# ============================================================
# RANDOM FOREST LAG MODEL
# ============================================================

def create_lag_features(
    series,
    lags=7
):

    data = pd.DataFrame(
        {"value": series.values}
    )

    for lag in range(1, lags + 1):

        data[f"lag_{lag}"] = (
            data["value"].shift(lag)
        )

    data = data.dropna()

    return data


def random_forest_forecast(
    train,
    horizon,
    lags=7
):

    data = create_lag_features(
        train,
        lags
    )

    X = data.drop(
        columns=["value"]
    )

    y = data["value"]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X,
        y
    )

    history = list(train.values)

    predictions = []

    for _ in range(horizon):

        features = np.array(
            history[-lags:]
        ).reshape(1, -1)

        prediction = model.predict(
            features
        )[0]

        predictions.append(
            prediction
        )

        history.append(
            prediction
        )

    return np.array(predictions), model


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌦️ Weather Analytics")

page = st.sidebar.radio(
    "Select Analysis",
    [
        "Dashboard",
        "Data Quality",
        "EDA",
        "Forecasting",
        "Anomaly Detection",
        "Environmental Impact",
        "Feature Importance",
        "Spatial Analysis",
        "Geographical Patterns"
    ]
)


# ============================================================
# LOAD DATA
# ============================================================

try:

    raw_df = load_data()

except Exception as e:

    st.error(str(e))
    st.stop()


columns = detect_columns(
    raw_df
)

df = preprocess_data(
    raw_df,
    columns
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌦️ Global Weather Trend & Forecasting</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Interactive weather analytics, forecasting, anomaly detection, '
    'environmental analysis and geographical patterns.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📊 Weather Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Observations",
            f"{len(df):,}"
        )

    with col2:

        if columns["country"]:

            st.metric(
                "Countries",
                df[columns["country"]].nunique()
            )

    with col3:

        if columns["location"]:

            st.metric(
                "Locations",
                df[columns["location"]].nunique()
            )

    with col4:

        if columns["temperature"]:

            st.metric(
                "Avg Temperature",
                f"{df[columns['temperature']].mean():.2f} °C"
            )

    st.divider()

    if columns["date"]:

        min_date = df[columns["date"]].min()
        max_date = df[columns["date"]].max()

        st.write(
            f"**Observation period:** "
            f"{min_date.date()} → {max_date.date()}"
        )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.subheader("Available Columns")

    column_table = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isna().sum().values
        }
    )

    st.dataframe(
        column_table,
        use_container_width=True
    )


# ============================================================
# DATA QUALITY
# ============================================================

elif page == "Data Quality":

    st.header("🧹 Data Cleaning & Quality")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Rows",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Columns",
            len(df.columns)
        )

    with col3:

        st.metric(
            "Duplicate Rows",
            f"{df.duplicated().sum():,}"
        )

    st.subheader("Missing Values")

    missing = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing_df = pd.DataFrame(
        {
            "Column": missing.index,
            "Missing Values": missing.values,
            "Missing %": (
                missing.values /
                len(df) * 100
            )
        }
    )

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    st.subheader("Numerical Summary")

    st.dataframe(
        df.describe().T,
        use_container_width=True
    )

    st.subheader("Outlier Detection")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    selected_column = st.selectbox(
        "Select numerical variable",
        numeric_columns
    )

    q1 = df[selected_column].quantile(0.25)
    q3 = df[selected_column].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df[
        (df[selected_column] < lower) |
        (df[selected_column] > upper)
    ]

    st.write(
        f"Lower bound: **{lower:.3f}**"
    )

    st.write(
        f"Upper bound: **{upper:.3f}**"
    )

    st.write(
        f"Detected statistical outliers: "
        f"**{len(outliers):,}**"
    )

    fig = px.box(
        df,
        y=selected_column,
        title=f"Boxplot - {selected_column}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# EDA
# ============================================================

elif page == "EDA":

    st.header("📈 Exploratory Data Analysis")

    if columns["date"]:

        date_col = columns["date"]

        st.subheader(
            "Weather Trend Over Time"
        )

        if columns["temperature"]:

            daily_temp = (
                df.set_index(date_col)
                [columns["temperature"]]
                .resample("D")
                .mean()
                .reset_index()
            )

            fig = px.line(
                daily_temp,
                x=date_col,
                y=columns["temperature"],
                title="Daily Average Temperature"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        if columns["precipitation"]:

            daily_precip = (
                df.set_index(date_col)
                [columns["precipitation"]]
                .resample("D")
                .sum()
                .reset_index()
            )

            fig = px.line(
                daily_precip,
                x=date_col,
                y=columns["precipitation"],
                title="Daily Precipitation"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.subheader("Correlation Analysis")

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.shape[1] >= 2:

        corr = numeric_df.corr(
            method="spearman"
        )

        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Spearman Correlation Matrix"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if columns["temperature"]:

        st.subheader(
            "Temperature Distribution"
        )

        fig = px.histogram(
            df,
            x=columns["temperature"],
            nbins=50,
            title="Temperature Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if columns["precipitation"]:

        st.subheader(
            "Precipitation Distribution"
        )

        fig = px.histogram(
            df,
            x=columns["precipitation"],
            nbins=50,
            title="Precipitation Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if columns["country"] and columns["temperature"]:

        st.subheader(
            "Average Temperature by Country"
        )

        country_temp = (
            df.groupby(columns["country"])[
                columns["temperature"]
            ]
            .mean()
            .sort_values(
                ascending=False
            )
            .head(30)
            .reset_index()
        )

        fig = px.bar(
            country_temp,
            x=columns["temperature"],
            y=columns["country"],
            orientation="h",
            title="Top Countries by Average Temperature"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# FORECASTING
# ============================================================

elif page == "Forecasting":

    st.header("🔮 Weather Forecasting")

    if not columns["date"]:

        st.error(
            "A date/time column is required for forecasting."
        )
        st.stop()

    target_options = []

    if columns["temperature"]:
        target_options.append(
            columns["temperature"]
        )

    if columns["precipitation"]:
        target_options.append(
            columns["precipitation"]
        )

    if not target_options:

        st.error(
            "Temperature or precipitation column not detected."
        )
        st.stop()

    target = st.selectbox(
        "Forecast Variable",
        target_options
    )

    if columns["location"]:

        locations = sorted(
            df[columns["location"]]
            .dropna()
            .astype(str)
            .unique()
        )

        selected_location = st.selectbox(
            "Location",
            locations
        )

        filtered = df[
            df[columns["location"]].astype(str)
            == selected_location
        ].copy()

    else:

        filtered = df.copy()

    filtered = filtered.sort_values(
        columns["date"]
    )

    # Daily aggregation
    series = (
        filtered
        .set_index(columns["date"])[target]
        .resample("D")
        .mean()
        .dropna()
    )

    if len(series) < 60:

        st.warning(
            "Not enough daily observations for reliable forecasting."
        )
        st.stop()

    horizon = st.slider(
        "Forecast Horizon (days)",
        min_value=7,
        max_value=60,
        value=30
    )

    if len(series) <= horizon + 30:

        st.warning(
            "The selected location has limited data."
        )
        st.stop()

    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]

    st.subheader(
        f"Forecasting {target}"
    )

    st.write(
        f"Training observations: **{len(train)}**"
    )

    st.write(
        f"Testing observations: **{len(test)}**"
    )

    # -----------------------------
    # Seasonal Naive
    # -----------------------------

    try:

        seasonal_prediction = (
            seasonal_naive_forecast(
                train,
                horizon
            )
        )

        seasonal_metrics = calculate_metrics(
            test,
            seasonal_prediction
        )

    except Exception:

        seasonal_prediction = None
        seasonal_metrics = None

    # -----------------------------
    # Holt Winters
    # -----------------------------

    try:

        hw_prediction = (
            holt_winters_forecast(
                train,
                horizon
            )
        )

        hw_metrics = calculate_metrics(
            test,
            hw_prediction
        )

    except Exception:

        hw_prediction = None
        hw_metrics = None

    # -----------------------------
    # Random Forest
    # -----------------------------

    try:

        rf_prediction, rf_model = (
            random_forest_forecast(
                train,
                horizon
            )
        )

        rf_metrics = calculate_metrics(
            test,
            rf_prediction
        )

    except Exception:

        rf_prediction = None
        rf_model = None
        rf_metrics = None

    # -----------------------------
    # Model comparison
    # -----------------------------

    results = []

    if seasonal_metrics:

        results.append(
            {
                "Model": "Seasonal Naive",
                **seasonal_metrics
            }
        )

    if hw_metrics:

        results.append(
            {
                "Model": "Holt-Winters",
                **hw_metrics
            }
        )

    if rf_metrics:

        results.append(
            {
                "Model": "Random Forest",
                **rf_metrics
            }
        )

    results_df = pd.DataFrame(
        results
    )

    st.subheader(
        "Model Comparison"
    )

    st.dataframe(
        results_df.style.format(
            {
                "MAE": "{:.3f}",
                "RMSE": "{:.3f}",
                "MAPE (%)": "{:.2f}",
                "R²": "{:.3f}"
            }
        ),
        use_container_width=True
    )

    # -----------------------------
    # Forecast visualization
    # -----------------------------

    forecast_dates = test.index

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=train.index,
            y=train.values,
            name="Training"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=test.index,
            y=test.values,
            name="Actual"
        )
    )

    if seasonal_prediction is not None:

        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=seasonal_prediction,
                name="Seasonal Naive"
            )
        )

    if hw_prediction is not None:

        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=hw_prediction,
                name="Holt-Winters"
            )
        )

    if rf_prediction is not None:

        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=rf_prediction,
                name="Random Forest"
            )
        )

    fig.update_layout(
        title="Forecast Comparison",
        xaxis_title="Date",
        yaxis_title=target
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------
    # Ensemble
    # -----------------------------

    predictions = []

    if seasonal_prediction is not None:
        predictions.append(
            seasonal_prediction
        )

    if hw_prediction is not None:
        predictions.append(
            hw_prediction
        )

    if rf_prediction is not None:
        predictions.append(
            rf_prediction
        )

    if len(predictions) >= 2:

        ensemble = np.mean(
            predictions,
            axis=0
        )

        ensemble_metrics = calculate_metrics(
            test,
            ensemble
        )

        st.subheader(
            "Ensemble Forecast"
        )

        st.write(
            "The ensemble combines the available "
            "forecasting models using equal weights."
        )

        ensemble_df = pd.DataFrame(
            {
                "Actual": test.values,
                "Ensemble": ensemble
            },
            index=test.index
        )

        fig = px.line(
            ensemble_df,
            title="Actual vs Ensemble Forecast"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.write(
            "Ensemble Metrics"
        )

        st.dataframe(
            pd.DataFrame(
                [ensemble_metrics]
            ).style.format(
                {
                    "MAE": "{:.3f}",
                    "RMSE": "{:.3f}",
                    "MAPE (%)": "{:.2f}",
                    "R²": "{:.3f}"
                }
            ),
            use_container_width=True
        )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif page == "Anomaly Detection":

    st.header("🚨 Weather Anomaly Detection")

    numeric_options = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    selected_variable = st.selectbox(
        "Variable",
        numeric_options
    )

    contamination = st.slider(
        "Expected anomaly proportion",
        min_value=0.001,
        max_value=0.10,
        value=0.02
    )

    anomaly_df = df[
        [selected_variable]
    ].copy()

    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    anomaly_df["anomaly"] = model.fit_predict(
        anomaly_df[[selected_variable]]
    )

    anomaly_df["anomaly_label"] = np.where(
        anomaly_df["anomaly"] == -1,
        "Anomaly",
        "Normal"
    )

    anomaly_count = (
        anomaly_df["anomaly"] == -1
    ).sum()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Total Observations",
            f"{len(anomaly_df):,}"
        )

    with col2:

        st.metric(
            "Detected Anomalies",
            f"{anomaly_count:,}"
        )

    fig = px.scatter(
        anomaly_df.reset_index(),
        x="index",
        y=selected_variable,
        color="anomaly_label",
        title=f"Anomalies in {selected_variable}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Detected Anomalies"
    )

    st.dataframe(
        anomaly_df[
            anomaly_df["anomaly"] == -1
        ],
        use_container_width=True
    )


# ============================================================
# ENVIRONMENTAL IMPACT
# ============================================================

elif page == "Environmental Impact":

    st.header("🌱 Environmental Impact Analysis")

    air_quality_columns = []

    if columns["pm25"]:
        air_quality_columns.append(
            columns["pm25"]
        )

    if columns["pm10"]:
        air_quality_columns.append(
            columns["pm10"]
        )

    if not air_quality_columns:

        st.warning(
            "No PM2.5 or PM10 column was detected."
        )
        st.stop()

    weather_columns = []

    for key in [
        "temperature",
        "precipitation",
        "humidity",
        "wind"
    ]:

        if columns[key]:
            weather_columns.append(
                columns[key]
            )

    selected_air = st.selectbox(
        "Air Quality Variable",
        air_quality_columns
    )

    selected_weather = st.selectbox(
        "Weather Variable",
        weather_columns
    )

    temp_df = df[
        [
            selected_air,
            selected_weather
        ]
    ].dropna()

    correlation = temp_df[
        selected_air
    ].corr(
        temp_df[selected_weather],
        method="spearman"
    )

    st.metric(
        "Spearman Correlation",
        f"{correlation:.3f}"
    )

    fig = px.scatter(
        temp_df.sample(
            min(5000, len(temp_df)),
            random_state=42
        ),
        x=selected_weather,
        y=selected_air,
        trendline="ols",
        title=(
            f"{selected_weather} vs "
            f"{selected_air}"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Environmental Correlation Matrix"
    )

    environmental_columns = (
        air_quality_columns +
        weather_columns
    )

    corr = df[
        environmental_columns
    ].corr(
        method="spearman"
    )

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Weather & Air Quality Correlations"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif page == "Feature Importance":

    st.header("⭐ Feature Importance")

    if not columns["temperature"]:

        st.warning(
            "Temperature column not detected."
        )
        st.stop()

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    target = columns["temperature"]

    feature_columns = [
        col
        for col in numeric_columns
        if col != target
    ]

    if len(feature_columns) == 0:

        st.warning(
            "Not enough numerical features."
        )
        st.stop()

    feature_df = df[
        feature_columns + [target]
    ].dropna()

    X = feature_df[
        feature_columns
    ]

    y = feature_df[
        target
    ]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X,
        y
    )

    importance = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Importance": model.feature_importances_
        }
    ).sort_values(
        "Importance",
        ascending=False
    )

    fig = px.bar(
        importance.head(20),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        importance,
        use_container_width=True
    )

    csv = importance.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Feature Importance",
        csv,
        "feature_importance.csv",
        "text/csv"
    )


# ============================================================
# SPATIAL ANALYSIS
# ============================================================

elif page == "Spatial Analysis":

    st.header("🗺️ Spatial Weather Analysis")

    if not columns["latitude"] or not columns["longitude"]:

        st.warning(
            "Latitude and longitude columns "
            "were not detected."
        )
        st.stop()

    map_variable_options = []

    for key in [
        "temperature",
        "precipitation",
        "humidity",
        "pm25"
    ]:

        if columns[key]:

            map_variable_options.append(
                columns[key]
            )

    if not map_variable_options:

        st.warning(
            "No weather variable is available."
        )
        st.stop()

    selected_variable = st.selectbox(
        "Map Variable",
        map_variable_options
    )

    map_df = df[
        [
            columns["latitude"],
            columns["longitude"],
            selected_variable
        ]
    ].dropna()

    # Limit points for browser performance
    if len(map_df) > 10000:

        map_df = map_df.sample(
            10000,
            random_state=42
        )

    # Ensure map coordinates and selected variable are numeric.
    map_df = map_df.copy()
    map_df[columns["latitude"]] = pd.to_numeric(
        map_df[columns["latitude"]], errors="coerce"
    )
    map_df[columns["longitude"]] = pd.to_numeric(
        map_df[columns["longitude"]], errors="coerce"
    )
    map_df[selected_variable] = pd.to_numeric(
        map_df[selected_variable], errors="coerce"
    )

    map_df = map_df.dropna(
        subset=[
            columns["latitude"],
            columns["longitude"],
            selected_variable
        ]
    )

    # Keep only valid geographic coordinates.
    map_df = map_df[
        map_df[columns["latitude"]].between(-90, 90)
        & map_df[columns["longitude"]].between(-180, 180)
    ]

    if map_df.empty:
        st.warning(
            "No valid latitude/longitude records are available for the map."
        )
        st.stop()

    # Plotly 5.24+ uses MapLibre and px.scatter_map.
    # Older Plotly versions use px.scatter_mapbox.
    # Supporting both makes the app work locally and on Streamlit Cloud.
    if hasattr(px, "scatter_map"):
        fig = px.scatter_map(
            map_df,
            lat=columns["latitude"],
            lon=columns["longitude"],
            color=selected_variable,
            zoom=1,
            height=600,
            title=f"Geographical Distribution of {selected_variable}",
            map_style="open-street-map"
        )
    else:
        fig = px.scatter_mapbox(
            map_df,
            lat=columns["latitude"],
            lon=columns["longitude"],
            color=selected_variable,
            zoom=1,
            height=600,
            title=f"Geographical Distribution of {selected_variable}"
        )

        fig.update_layout(
            mapbox_style="open-street-map"
        )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# GEOGRAPHICAL PATTERNS
# ============================================================

elif page == "Geographical Patterns":

    st.header("🌍 Geographical Weather Patterns")

    if not columns["country"]:

        st.warning(
            "Country column not detected."
        )
        st.stop()

    metric_options = []

    for key in [
        "temperature",
        "precipitation",
        "humidity",
        "pm25"
    ]:

        if columns[key]:

            metric_options.append(
                columns[key]
            )

    selected_metric = st.selectbox(
        "Metric",
        metric_options
    )

    country_summary = (
        df.groupby(columns["country"])[
            selected_metric
        ]
        .agg(
            [
                "mean",
                "median",
                "std",
                "min",
                "max"
            ]
        )
        .reset_index()
    )

    country_summary = country_summary.sort_values(
        "mean",
        ascending=False
    )

    st.subheader(
        "Country-level Summary"
    )

    st.dataframe(
        country_summary,
        use_container_width=True
    )

    fig = px.bar(
        country_summary.head(30),
        x="mean",
        y=columns["country"],
        orientation="h",
        title=(
            f"Top 30 Countries by "
            f"Average {selected_metric}"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    csv = country_summary.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Country Summary",
        csv,
        "country_weather_summary.csv",
        "text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Global Weather Trend & Forecasting | "
    "Data Science & Machine Learning Project"
)
