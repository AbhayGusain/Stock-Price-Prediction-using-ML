

# Stock Price Prediction using Machine Learning

Predict stock prices using machine learning and deep learning models. This project provides a web app and a Jupyter notebook for data analysis, model training, and prediction.

## Project Overview
This project uses historical stock data to train a neural network model (LSTM) for predicting future stock prices. It includes:
- Data collection from Yahoo Finance
- Data preprocessing and visualization
- Model training and evaluation
- Interactive web app for predictions

## Files
- `app.py`: Streamlit web app for interactive predictions
- `Stock Predictions Model.keras`: Trained Keras LSTM model
- `Stock_Market_Prediction_Model_Creation.ipynb`: Jupyter notebook for data analysis and model creation
- `requirements.txt`: List of required Python packages

## Example Usage
### Using the Web App
1. Run the app:
	```powershell
	streamlit run app.py
	```
2. Enter a stock symbol (e.g., GOOG, AAPL, MSFT) in the input box.
3. View historical data, moving averages, and predicted prices.

### Using the Notebook
1. Open `Stock_Market_Prediction_Model_Creation.ipynb` in Jupyter Notebook:
	```powershell
	jupyter notebook Stock_Market_Prediction_Model_Creation.ipynb
	```
2. Run the cells to see data analysis, model training, and predictions step-by-step.

## How It Works
1. **Data Collection:** Fetches historical stock prices using `yfinance`.
2. **Preprocessing:** Scales and splits data for training/testing.
3. **Model Training:** Builds and trains an LSTM neural network using Keras.
4. **Prediction:** Predicts future prices and visualizes results.
5. **Web App:** Lets users input a stock symbol and see predictions interactively.

## Steps to Run This Project
1. **Clone the repository:**
	```powershell
	git clone https://github.com/AbhayGusain/Stock-Price-Prediction-using-ML.git
	cd Stock-Price-Prediction-using-ML
	```
2. **Install dependencies:**
	```powershell
	pip install -r requirements.txt
	```
3. **Run the web app:**
	```powershell
	streamlit run app.py
	```
4. **Or run the notebook:**
	```powershell
	jupyter notebook Stock_Market_Prediction_Model_Creation.ipynb
	```

## Requirements
- Python 3.x
- See `requirements.txt` for all required packages

## License
MIT
