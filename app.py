import streamlit as st
import pandas as pd
import datetime
import pickle
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from stock_modelling import load_data, preprocess_data
from sklearn.preprocessing import LabelEncoder

st.set_page_config(layout="wide")

def load_model(filename=r'pickle_files/model_stock.pkl',
               scaler_filename=r'pickle_files/scaler_stock.pkl'):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    with open(scaler_filename, 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

def create_plots(subset_df, stock_name, selected_days, chart_type):
    if chart_type == "Candlestick":
        fig = go.Figure(data=[go.Candlestick(x=subset_df.index,
                                             open=subset_df['Open'],
                                             high=subset_df['High'],
                                             low=subset_df['Low'],
                                             close=subset_df['Close'])])
        fig.update_layout(title=f"{stock_name} - Closing Prices for the last {selected_days} days",
                          xaxis_title="Date",
                          yaxis_title="Closing Price",
                          xaxis_rangeslider_visible=False,
                          template="plotly_white")
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(subset_df.index, subset_df['Close'], label=stock_name, color='cyan')
        ax.set_title(f"{stock_name} - Closing Prices for the last {selected_days} days")
        ax.set_xlabel("Date")
        ax.set_ylabel("Closing Price")
        ax.legend()
        plt.xticks(rotation=45)
    return fig

def main():
    stocks = {
        'AAPL': 'Apple Inc',
        'AMZN': 'Amazon.com Inc',
        'GOOG': 'Alphabet Inc Class C',
        'META': 'Meta Platforms Inc'
    }

    start_date = '2021-08-01'
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    stock_df = load_data(list(stocks.keys()), start_date, end_date)
    stock_df = preprocess_data(stock_df)
    model, scaler = load_model()

    st.title("Stock Price Prediction")
    st.image('https://assets.cmcmarkets.com/images/Forex-vs-stocks.jpg', caption='Stock Analysis Dashboard',
             use_column_width=False)
    st.write("Forecasting Closing Price of Stocks")
    st.write('Developed by: Onibuje .A. Olalekan')

    st.sidebar.header("Stocks")
    stock_name = st.sidebar.selectbox("Select Stock", [f"{symbol} ({name})" for symbol, name in stocks.items()])
    selected_symbol = stock_name.split(" ")[0]  # Extracts the symbol from the selected option

    if st.button('Predict'):
        last_info = stock_df[stock_df['currency'] == selected_symbol].tail(1).drop(columns=['Adj Close', 'Close'])
        le = LabelEncoder()
        last_info['currency'] = le.fit_transform([selected_symbol])[0]
        last_info_scaled = scaler.transform(last_info)
        predicted_price = model.predict(last_info_scaled)

        # Display the selected stock with its company name
        stock_name_display = f"{selected_symbol} ({stocks[selected_symbol]})"
        st.write(f"Selected Stock: {stock_name_display}")
        st.write(f"Predicted next closing price for {datetime.datetime.now().strftime('%Y-%m-%d')}: ${predicted_price[0]:.2f}")

    st.sidebar.subheader("Visualization")
    chart_type = st.sidebar.radio("Select Chart Type", ["Candlestick", "Line"])
    st.sidebar.header("Select Period")
    period = st.sidebar.selectbox("Period", ["7 days", "30 days", "60 days", "1 year"])

    days_map = {"7 days": 7, "30 days": 30, "60 days": 60, "1 year": 365}
    selected_days = days_map[period]

    st.write("------------------------------------------------------------------------------------------------------------")

    if selected_symbol:
        st.write(f"SHOWING DATA FOR THE LAST {selected_days} DAYS FOR {selected_symbol}")
        subset_df = stock_df[stock_df['currency'] == selected_symbol].tail(selected_days)
        if subset_df.empty:
            st.error(f"No data available for the last {selected_days} days for {selected_symbol}.")
        else:
            fig = create_plots(subset_df, selected_symbol, selected_days, chart_type)
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
