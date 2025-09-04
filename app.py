
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import os
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(page_title="Stock Market Predictor", layout="centered")
# Set background color to white using custom CSS
st.markdown(
    """
    <style>
        .stApp {
            background-color: white !important;
            color: black !important;
        }
        /* Sidebar background and text color */
        section[data-testid="stSidebar"] {
            background-color: white !important;
            color: black !important;
        }
        section[data-testid="stSidebar"] * {
            color: black !important;
        }
        h1, h2, h3, h4, h5, h6, .stMarkdown, .stTextInput label, .stSubheader, .stHeader, .stCaption, .stDataFrame, .stTable, .stText, .stInfo, .stWarning, .stError {
            color: black !important;
        }
        .css-1d391kg, .css-1v0mbdj, .css-1cpxqw2, .css-1lcbmhc, .css-1y4p8pa, .css-1v3fvcr {
            color: black !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Stock Market Predictor')
st.markdown("""
Enter a valid stock symbol (e.g., GOOG, AAPL, MSFT) to view historical data and predict future prices using a trained ML model.""")

model_path = os.path.join(os.path.dirname(__file__), 'Stock Predictions Model.keras')
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


stock = st.text_input('Enter Stock Symbol', 'GOOG').upper().strip()
start = '2020-01-01'
end = '2025-08-31'

if stock:
    try:
        data = yf.download(stock, start, end)
        if data.empty:
            st.error(f"No data found for symbol '{stock}'. Please enter a valid stock symbol.")
        else:
            st.subheader('Stock Data')
            st.markdown("""
            **Explanation:**
            The table below shows the historical daily stock prices for your selected symbol. Key columns include Open, High, Low, Close, Volume, and Adjusted Close. This data is used for analysis and prediction.
            """)
            st.write(data)

            data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
            data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0,1))

            pas_100_days = data_train.tail(100)
            data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
            data_test_scale = scaler.fit_transform(data_test)

            st.subheader('Price vs MA50')
            st.markdown("""
            **Explanation:**
            This chart compares the actual closing price (green) with the 50-day moving average (red). Moving averages help smooth out price fluctuations and highlight trends over time.
            """)
            ma_50_days = data.Close.rolling(50).mean()
            fig1 = plt.figure(figsize=(8,6))
            plt.plot(data.Close, 'g', label='Close Price')
            plt.plot(ma_50_days, 'r', label='MA50')
            plt.legend()
            st.pyplot(fig1)

            st.subheader('Price vs MA50 vs MA100')
            st.markdown("""
            **Explanation:**
            This chart shows the closing price (green), 50-day moving average (red), and 100-day moving average (blue). Comparing multiple moving averages can help identify longer-term trends and potential buy/sell signals.
            """)
            ma_100_days = data.Close.rolling(100).mean()
            fig2 = plt.figure(figsize=(8,6))
            plt.plot(data.Close, 'g', label='Close Price')
            plt.plot(ma_50_days, 'r', label='MA50')
            plt.plot(ma_100_days, 'b', label='MA100')
            plt.legend()
            st.pyplot(fig2)

            st.subheader('Price vs MA100 vs MA200')
            st.markdown("""
            **Explanation:**
            This chart displays the closing price (green), 100-day moving average (red), and 200-day moving average (blue). The 200-day moving average is often used to assess long-term market direction.
            """)
            ma_200_days = data.Close.rolling(200).mean()
            fig3 = plt.figure(figsize=(8,6))
            plt.plot(data.Close, 'g', label='Close Price')
            plt.plot(ma_100_days, 'r', label='MA100')
            plt.plot(ma_200_days, 'b', label='MA200')
            plt.legend()
            st.pyplot(fig3)

            x = []
            y = []
            for i in range(100, data_test_scale.shape[0]):
                x.append(data_test_scale[i-100:i])
                y.append(data_test_scale[i,0])
            x, y = np.array(x), np.array(y)

            if x.shape[0] == 0:
                st.warning("Not enough data for prediction. Try a different stock or time range.")
            else:
                predict = model.predict(x)
                scale = 1/scaler.scale_
                predict = predict * scale
                y = y * scale

                st.subheader('Predicted Price vs Original Price')
                st.markdown("""
                **Explanation:**
                The chart below compares the actual stock prices (green) with the prices predicted by the machine learning model (red). This helps you see how well the model fits the real data and how it might perform in forecasting future prices.
                """)
                fig4 = plt.figure(figsize=(8,6))
                plt.plot(y, 'g', label='Original Price')
                plt.plot(predict, 'r', label='Predicted Price')
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.legend()
                st.pyplot(fig4)
    except Exception as e:
        st.error(f"Error fetching or processing data: {e}")
else:
    st.info("Please enter a stock symbol above.")

