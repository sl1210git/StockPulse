# STOCK PULSE - Stock Price Prediction Application

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Collection](#data-collection)
  - [Model Training](#model-training)
  - [Visualization](#visualization)
- [Technologies Used](#technologies-used)

## Introduction

STOCK PULSE can be used to forecasting the future value of a company's stock based on historical market data, technical indicators, and news sentiment analysis. The application leverages various machine learning models, such as Gradient Boosting, XGBoost, Support Vector, Decision Tree, Linear Regressor and CatBoost Regressor to predict stock prices accurately. It provides a comprehensive solution for predicting stock prices using machine learning models and data science techniques. The repository includes data collection from yfinance, preprocessing, model building, evaluation, visualization scripts and deployment.

## Features

- **Data Collection**: Fetch historical stock data, including prices, volumes, and other indicators, from Yahoo Finance.
- **Data Preprocessing**: Clean, normalize, and transform raw data to prepare it for model training.
- **Exploratory Data Analysis (EDA)**: Visualize and analyze historical stock trends and patterns.
- **Model Building**: Implement multiple machine learning models to predict future stock prices.
- **Backtesting and Evaluation**: Evaluate model performance using metrics like MSE and MAE.
- **Interactive Dashboard**: Visualize stock trends, predictions, and performance metrics using a Streamlit dashboard.

## Technologies Used

- **Python**: The core programming language.
- **Libraries**: 
  - Data Handling: `Pandas`, `NumPy`
  - Machine Learning: `scikit-learn`, `TensorFlow`, `PyTorch`, `Statsmodels`
  - Visualization: `Plotly`, `Matplotlib`
- **Tools**: PyCharm, Streamlit

## Installation

To get started, clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/Onibuje-Olalekan/stock-prediction.git
cd stock-prediction
pip install -r requirements.txt
brew install libomp
```

## Usage

1. Run the Streamlit app to visualize stock data and model predictions:
    ```bash
    streamlit run app.py
    ```
2. Interact with the dashboard to explore stock trends and evaluate model performance.