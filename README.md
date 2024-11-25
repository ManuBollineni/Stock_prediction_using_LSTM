# Stock Price Prediction with LSTM

This project demonstrates how to use an **LSTM (Long Short-Term Memory)** model to predict stock prices based on historical stock data. The model is trained using **TensorFlow** and **Keras**, and the predictions can be accessed through a **Streamlit** web application.

## Project Overview

The goal of this project is to predict future stock prices using a **recurrent neural network** (RNN) based on the LSTM architecture. We use historical stock price data (including additional features such as **RSI** and **EMA**) to train the model and then deploy it as a web application for users to interact with.

### Key Features

- **Stock Price Prediction** using LSTM model.
- **Interactive Web Interface** built with **Streamlit**.
- Data sourced from **Yahoo Finance**.
- Features include stock price, volume, **RSI** (Relative Strength Index), and **EMA** (Exponential Moving Average).
- **Model Training** and **Prediction** using **TensorFlow/Keras**.

## Project Structure

The repository contains the following key files:

- `lstm_stock_model.h5`: Pre-trained LSTM model saved after training.
- `stock_prediction_app.py`: Streamlit web app for making stock predictions.
- `Stock_Price_Prediction_Training.ipynb`: Jupyter notebook used for model training and evaluation.
- `requirements.txt`: List of dependencies required for the project.

## Requirements

### Python 3.x

You need to install the following libraries to run this project. You can install them using **pip**.

- **TensorFlow**: For building and training the LSTM model.
- **Streamlit**: To create the interactive web application.
- **yfinance**: To download historical stock data from Yahoo Finance.
- **ta**: For calculating technical indicators like RSI and EMA.
- **Matplotlib**: For plotting graphs.
- **scikit-learn**: For data preprocessing and scaling.

You can install these dependencies by running the following command:

```bash
pip install -r requirements.txt
```

Alternatively, you can install them individually:

```bash
pip install tensorflow streamlit yfinance ta matplotlib scikit-learn
```

### Running the Web App

1. Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/stock-price-prediction-lstm.git
cd stock-price-prediction-lstm
```

2. Run the Streamlit web app:

```bash
streamlit run stock_prediction_app.py
```

3. The app will open in your browser. You can enter a stock ticker (e.g., AAPL for Apple), specify a start date and end date, and view the predicted stock prices.

## Data Description

The data used for training and prediction is sourced from **Yahoo Finance**. The dataset includes the following columns:

- **Date**: Date of the stock data.
- **Open**: Opening price of the stock.
- **High**: Highest price of the stock during the day.
- **Low**: Lowest price of the stock during the day.
- **Close**: Closing price of the stock.
- **Adj Close**: Adjusted closing price (factoring in stock splits and dividends).
- **Volume**: Number of shares traded.
- **Return**: Daily return percentage based on adjusted close price.
- **RSI**: Relative Strength Index, a momentum indicator.
- **EMA**: Exponential Moving Average, a smoothing technique for stock prices.

## Model Architecture

The **LSTM model** used in this project has the following architecture:

1. **LSTM Layer** with 50 units and ReLU activation function.
2. **Dropout Layer** with 20% rate to prevent overfitting.
3. Another **LSTM Layer** with 50 units.
4. **Dropout Layer** with 20% rate.
5. **Dense Layer** with a single unit to output the predicted stock price.

The model is compiled using **Adam optimizer** and **Mean Squared Error (MSE)** as the loss function.

## Model Training

The model was trained using historical stock data from 2015 to 2023. A **lookback period** of 14 days was used to predict the stock price for the next day. The training dataset consisted of **80%** of the data, and **20%** was used for validation.

After training, the model was saved to a file (`lstm_stock_model.h5`) which can be used for future predictions.

## How the Web App Works

1. The user inputs the stock ticker, start date, and end date.
2. The app downloads the historical stock data using **yfinance** and calculates additional features like **RSI** and **EMA**.
3. The data is preprocessed, and the model is loaded to make predictions.
4. The **predicted stock prices** are plotted against the **actual prices** for visual comparison.

## Example Usage

1. Clone the repository and install dependencies as described above.
2. Run the Streamlit app:

```bash
streamlit run stock_prediction_app.py
```

3. Enter a stock ticker (e.g., AAPL) and specify the date range.
4. View the predicted stock prices and compare them to the actual stock prices.
