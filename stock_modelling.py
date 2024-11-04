import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler,MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

pd.set_option('display.float_format', '{:.2f}'.format)

def load_data(stocks, start_date, end_date):
    stock_df = pd.DataFrame()
    for ticker in stocks:
        data = yf.download(ticker, start=start_date, end=end_date)
        data['currency'] = ticker
        stock_df = pd.concat([stock_df, data])
    return stock_df

def preprocess_data(stock_df):
    stock_df.sort_values(by=['currency', 'Date'], ascending=True, inplace=True)

    # Creating lag features
    for lag in range(1, 8):  # Lag features from 1 to 7 days
        stock_df[f'lag_{lag}'] = stock_df.groupby('currency')['Close'].shift(lag)
    
    # Creating rolling statistics
    for lag in [7, 14, 21]:  # Rolling mean for the last 7, 14, and 21 days
        stock_df[f'rolling_{lag}'] = stock_df.groupby('currency')['Close'].shift(lag).rolling(lag).mean()

    # Feature extraction from datetime index
    stock_df['year'] = stock_df.index.year
    stock_df['month'] = stock_df.index.month
    stock_df['day'] = stock_df.index.day
    stock_df['quarter'] = stock_df.index.quarter
    stock_df['weekday'] = stock_df.index.weekday
    stock_df['is_weekend'] = (stock_df.index.weekday >= 5).astype(int)
    stock_df['is_start_of_month'] = (stock_df.index.day == 1).astype(int)
    stock_df['is_end_of_month'] = (stock_df.index.is_month_end).astype(int)

    # Drop rows with NaN values after creating features
    stock_df.dropna(inplace=True)

    return stock_df

def train_models(xtrain, ytrain):
    models = {
        # "Random Forest": RandomForestRegressor(n_estimators=500, random_state=0),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=500, random_state=0),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10, n_estimators=100),
        "Support Vector Regression": SVR(kernel='rbf'),
        "Decision Tree": DecisionTreeRegressor(random_state=0),
        "Linear Regressor": Lasso(),
        "CatBoostRegressor": CatBoostRegressor(verbose=0)
    }
    for model in models.values():
        model.fit(xtrain, ytrain)
    return models

def evaluate_models(models, xtest, ytest):
    results_list = []
    for name, model in models.items():
        y_pred = model.predict(xtest)
        mse = mean_squared_error(ytest, y_pred)
        mae = mean_absolute_error(ytest, y_pred)
        results_list.append([name, mse, mae])
    results_df = pd.DataFrame(results_list, columns=["Model", "MSE", "MAE"])
    return results_df.sort_values(by='MAE')

def save_model(model, scaler, filename=r'pickle_files/model_crypto.pkl', scaler_filename=r'pickle_files/scaler_crypto.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    with open(scaler_filename, 'wb') as file:
        pickle.dump(scaler, file)

def main():
    # Load and preprocess data
    stocks = ['AAPL', 'AMZN', 'GOOG', 'META']
    start_date = '2024-01-01'
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    stock_df = load_data(stocks, start_date, end_date)
    stock_df = preprocess_data(stock_df)
    
    # Prepare features and target
    # Shifting the target to be the current day's close price
    stock_df['target'] = stock_df.groupby('currency')['Close'].shift(-1)
    
    # Drop rows with NaN in the target column
    stock_df.dropna(subset=['target'], inplace=True)
    
    # Features and target variable
    X = stock_df.drop(columns=['Adj Close', 'Close', 'target'])
    le = LabelEncoder()
    X['currency'] = le.fit_transform(X['currency'])
    y = stock_df['target']  # Current day's close price

    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)

    # Train and evaluate models
    models = train_models(xtrain, ytrain)
    results_df = evaluate_models(models, xtest, ytest)
    print(results_df)

    # Save the best model
    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]
    save_model(best_model, scaler)

if __name__ == "__main__":
    main()
