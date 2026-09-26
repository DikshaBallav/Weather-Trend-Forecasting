## 🌦️ Weather Trend Forecasting & Climate Analysis

An end-to-end Data Science and Machine Learning project for analyzing global weather patterns, forecasting weather variables, detecting anomalies, and studying environmental and geographical relationships.

The project includes data preprocessing, exploratory data analysis, time-series forecasting, anomaly detection, feature importance, environmental analysis, spatial analysis, and an interactive Streamlit dashboard.

## 🚀 Live Streamlit Dashboard

🔗 **Live App:** [Open Weather Trend Forecasting Dashboard](https://weather-trend-forecasting-du4bxxxyqurhghhz72y4vf.streamlit.app/)

The interactive Streamlit dashboard provides access to:

- 📊 Exploratory Data Analysis  — interactive weather analysis
- 🔮 Weather Forecasting  — model comparison and predictions
- 🚨 Anomaly Detection - Isolation Forest results
- 🌱 Environmental Impact Analysis — air quality and weather relationships
- ⭐ Feature Importance — model-derived feature rankings
- 🌍 Spatial Analysis  — geographical visualizations
- 🗺️ Geographical Patterns  — country and continental comparisons
  
## 🎯 Objectives
Clean and preprocess historical weather data.
Perform exploratory data analysis and identify important trends and relationships.
Analyze temperature and precipitation patterns.
Build and compare multiple forecasting models.
Evaluate forecasting performance using multiple metrics.
Create an ensemble forecasting approach.
Detect anomalous weather observations.
Analyze relationships between air quality and weather parameters.
Identify important predictive features.
Study spatial and geographical weather patterns.
Compare weather conditions across countries and continents.
Provide an interactive dashboard for exploring the results.

## 📊 Dataset

The project uses GlobalWeatherRepository.csv, containing weather observations from multiple locations and countries.

The dataset includes variables related to:

Temperature
Humidity
Precipitation
Wind
Atmospheric pressure
Visibility
UV index
Air quality
Latitude and longitude
Country and location
Date/time

The raw CSV is not included in this repository because of its large file size.

To reproduce the project, place the dataset at:

data/
└── GlobalWeatherRepository.csv

Dataset Source: Add the original dataset URL here.

## 🔄 Project Workflow

Data Collection
      ↓
Data Cleaning & Preprocessing
      ↓
Exploratory Data Analysis
      ↓
Time-Series Analysis
      ↓
Forecasting Models
      ↓
Model Evaluation
      ↓
Ensemble Forecasting
      ↓
Anomaly Detection
      ↓
Environmental Analysis
      ↓
Feature Importance
      ↓
Spatial & Geographical Analysis
      ↓
Streamlit Dashboard

## 🧹 Data Cleaning & Preprocessing

The dataset is prepared through:

Missing-value analysis and handling
Duplicate detection and removal
Data type validation
Date/time conversion
Invalid-value checks
Outlier identification using IQR
Numerical feature scaling where required
Creation of time-based features such as year, month, and day

## 📈 Exploratory Data Analysis

The EDA focuses on:

Temperature distributions and trends
Precipitation patterns
Humidity and wind analysis
Correlation between weather variables
Temperature and precipitation relationships
Monthly and seasonal patterns
Country-level comparisons
Continental comparisons

Visualizations include line charts, bar charts, histograms, box plots, scatter plots, heatmaps, and geographical plots.

## 🔮 Forecasting & Model Evaluation

Multiple forecasting approaches are developed and compared:

Model	Purpose
Seasonal Naive	Baseline forecasting
Holt-Winters	Trend and seasonal forecasting
Random Forest + Lag Features	Machine-learning forecasting
Ensemble	Combines model predictions

Models are evaluated using:

MAE
RMSE
MAPE
R² where applicable

A chronological train/test split is used to preserve the time-series structure.

## 🚨 Anomaly Detection

Isolation Forest is used to identify potentially unusual weather observations.

The analysis includes:

Detection of anomalous observations
Anomaly distribution
Country/location analysis
Visualization of detected anomalies

Detected anomalies are treated as potentially unusual observations rather than automatically being classified as data errors.

## 🌱 Environmental Impact Analysis

The project investigates relationships between air-quality measurements and weather conditions.

Where available, variables such as PM2.5, PM10, CO, NO₂, SO₂, and O₃ are analyzed against weather parameters.

The analysis uses:

Correlation analysis
Spearman correlation
Scatter plots
Trend analysis
Geographical comparisons

## ⭐ Feature Importance

Tree-based machine learning models are used to identify important predictive features.

Feature importance is analyzed using:

Random Forest
Extra Trees

These results represent model-specific predictive importance and should not be interpreted as causal relationships.

## 🌍 Spatial & Geographical Analysis

Latitude, longitude, country, and location information are used to study geographical patterns.

The project analyzes:

Temperature by geographical location
Precipitation by region
Air-quality distribution
Country-level weather patterns
Continental differences
Monthly patterns across continents

Interactive geographical visualizations are available through Plotly.

📁 Repository Structure
Weather-Trend-Forecasting/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── Weather Trend Forecasting.ipynb
│
└── data/
    └── GlobalWeatherRepository.csv

The raw dataset is kept outside the public GitHub repository because of its size.

## ⚙️ Installation
1. Clone the repository
git clone https://github.com/DikshaBallav/Weather-Trend-Forecasting.git
cd Weather-Trend-Forecasting
2. Create a virtual environment
python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Add the dataset

Place:

GlobalWeatherRepository.csv

inside:

data/
5. Run the Streamlit application
streamlit run app.py

The application will open at:

http://localhost:8501

## 📓 Jupyter Notebook

The complete analysis is available in:

Weather Trend Forecasting.ipynb

The notebook contains the detailed implementation of:

Data preprocessing
EDA
Time-series analysis
Forecasting
Model evaluation
Anomaly detection
Environmental analysis
Feature importance
Spatial analysis
Geographical analysis
