# 📈 Stock Price Prediction using Machine Learning

A Machine Learning based web application for predicting stock prices using historical market data.

## 🚀 Live Project

This project is deployed using Streamlit Community Cloud.

## 📌 Project Overview

The application uses historical stock market data and Machine Learning algorithms to predict the next closing price of selected stocks.

Users can select a company and view historical prices, moving averages, model predictions, and model performance.

## 🛠️ Technologies Used

- Python
- Streamlit
- NumPy
- Pandas
- yFinance
- Scikit-learn
- Machine Learning

## 🤖 Machine Learning Models

The project currently uses:

- Linear Regression
- Random Forest Regressor

The application compares model performance using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

The model with better performance is selected for the final prediction.

## ✨ Features

- 📊 Stock price prediction
- 🏢 Multiple companies
- 📈 Historical stock price visualization
- 📉 7-day and 30-day moving averages
- 🤖 Actual vs predicted price comparison
- 🧪 Model performance comparison
- 💾 Prediction results CSV download
- 🌐 Interactive Streamlit dashboard

## 📁 Project Structure

```text
Stock-Price-Prediction/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
