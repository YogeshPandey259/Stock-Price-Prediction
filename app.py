import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)


# ==========================================================
# TITLE
# ==========================================================

st.title("📈 Stock Price Prediction")

st.write(
    "A Machine Learning based application for predicting "
    "the next closing price of selected stocks."
)


# ==========================================================
# COMPANY LIST
# ==========================================================

companies = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "NVIDIA": "NVDA",
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "ITC": "ITC.NS"
}


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("⚙️ Project Settings")

company_name = st.sidebar.selectbox(
    "Select Company",
    list(companies.keys())
)

ticker = companies[company_name]

st.sidebar.write("Ticker:", ticker)

start_date = st.sidebar.date_input(
    "Start Date",
    pd.to_datetime("2015-01-01")
)

end_date = st.sidebar.date_input(
    "End Date",
    pd.to_datetime("2026-01-01")
)


# ==========================================================
# PREDICTION BUTTON
# ==========================================================

if st.sidebar.button("🚀 Predict Stock Price"):

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    with st.spinner("Downloading data and training ML models..."):

        # --------------------------------------------------
        # DOWNLOAD DATA
        # --------------------------------------------------

        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True
        )

        if data.empty:
            st.error("No stock data found.")
            st.stop()


        # --------------------------------------------------
        # FIX MULTIINDEX
        # --------------------------------------------------

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data[
            ["Open", "High", "Low", "Close", "Volume"]
        ]

        data = data.dropna()


        # --------------------------------------------------
        # FEATURE ENGINEERING
        # --------------------------------------------------

        data["Previous_Close"] = data["Close"].shift(1)
        data["Previous_Open"] = data["Open"].shift(1)
        data["Previous_High"] = data["High"].shift(1)
        data["Previous_Low"] = data["Low"].shift(1)
        data["Previous_Volume"] = data["Volume"].shift(1)

        data["MA_7"] = data["Close"].rolling(7).mean()
        data["MA_30"] = data["Close"].rolling(30).mean()

        data["Daily_Return"] = data["Close"].pct_change()

        data["Target"] = data["Close"].shift(-1)

        data = data.dropna()


        # --------------------------------------------------
        # FEATURES
        # --------------------------------------------------

        features = [
            "Previous_Close",
            "Previous_Open",
            "Previous_High",
            "Previous_Low",
            "Previous_Volume",
            "MA_7",
            "MA_30",
            "Daily_Return"
        ]

        X = data[features]
        y = data["Target"]


        # --------------------------------------------------
        # TRAIN TEST SPLIT
        # --------------------------------------------------

        split = int(len(data) * 0.80)

        X_train = X.iloc[:split]
        X_test = X.iloc[split:]

        y_train = y.iloc[:split]
        y_test = y.iloc[split:]


        # ==================================================
        # LINEAR REGRESSION
        # ==================================================

        linear_model = LinearRegression()

        linear_model.fit(
            X_train,
            y_train
        )

        linear_predictions = linear_model.predict(
            X_test
        )


        # ==================================================
        # RANDOM FOREST
        # ==================================================

        rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        rf_model.fit(
            X_train,
            y_train
        )

        rf_predictions = rf_model.predict(
            X_test
        )


        # ==================================================
        # MODEL EVALUATION
        # ==================================================

        linear_mae = mean_absolute_error(
            y_test,
            linear_predictions
        )

        linear_rmse = np.sqrt(
            mean_squared_error(
                y_test,
                linear_predictions
            )
        )

        rf_mae = mean_absolute_error(
            y_test,
            rf_predictions
        )

        rf_rmse = np.sqrt(
            mean_squared_error(
                y_test,
                rf_predictions
            )
        )


        # ==================================================
        # BEST MODEL
        # ==================================================

        if rf_mae < linear_mae:

            best_model = rf_model
            best_predictions = rf_predictions
            best_model_name = "Random Forest"
            best_mae = rf_mae
            best_rmse = rf_rmse

        else:

            best_model = linear_model
            best_predictions = linear_predictions
            best_model_name = "Linear Regression"
            best_mae = linear_mae
            best_rmse = linear_rmse


        # ==================================================
        # NEXT DAY PREDICTION
        # ==================================================

        latest = data.iloc[-1]

        latest_features = pd.DataFrame(
            [[
                latest["Close"],
                latest["Open"],
                latest["High"],
                latest["Low"],
                latest["Volume"],
                latest["MA_7"],
                latest["MA_30"],
                latest["Daily_Return"]
            ]],
            columns=features
        )

        prediction = best_model.predict(
            latest_features
        )[0]

        current_price = float(latest["Close"])


        # ==================================================
        # PREDICTION CHANGE
        # ==================================================

        price_change = prediction - current_price

        price_change_percent = (
            price_change / current_price
        ) * 100


    # ======================================================
    # RESULTS
    # ======================================================

    st.success("✅ Prediction completed successfully!")

    st.header(
        f"📊 {company_name} ({ticker})"
    )


    # ======================================================
    # METRICS
    # ======================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Current Price",
            f"{current_price:.2f}"
        )

    with col2:

        st.metric(
            "Predicted Price",
            f"{prediction:.2f}"
        )

    with col3:

        st.metric(
            "Expected Change",
            f"{price_change_percent:.2f}%"
        )

    with col4:

        st.metric(
            "Best Model",
            best_model_name
        )


    # ======================================================
    # DIRECTION
    # ======================================================

    if prediction > current_price:

        st.success(
            "📈 Model prediction: Price may increase."
        )

    elif prediction < current_price:

        st.warning(
            "📉 Model prediction: Price may decrease."
        )

    else:

        st.info(
            "➡️ Model prediction: Price may remain approximately stable."
        )


    # ======================================================
    # HISTORICAL DATA
    # ======================================================

    st.subheader("📈 Historical Stock Price")

    chart_data = data[
        ["Close", "MA_7", "MA_30"]
    ].copy()

    chart_data.columns = [
        "Closing Price",
        "7 Day Moving Average",
        "30 Day Moving Average"
    ]

    st.line_chart(chart_data)


    # ======================================================
    # ACTUAL VS PREDICTED
    # ======================================================

    st.subheader("🤖 Actual vs Predicted Price")

    comparison = pd.DataFrame({
        "Actual Price": y_test.values,
        "Predicted Price": best_predictions
    })

    st.line_chart(comparison)


    # ======================================================
    # MODEL PERFORMANCE
    # ======================================================

    st.subheader("🧪 Model Performance")

    performance = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Random Forest"
        ],
        "MAE": [
            linear_mae,
            rf_mae
        ],
        "RMSE": [
            linear_rmse,
            rf_rmse
        ]
    })

    st.dataframe(
        performance,
        use_container_width=True
    )


    # ======================================================
    # LATEST STOCK DATA
    # ======================================================

    st.subheader("📋 Latest Stock Data")

    latest_table = data[
        ["Open", "High", "Low", "Close", "Volume"]
    ].tail(10)

    st.dataframe(
        latest_table,
        use_container_width=True
    )


    # ======================================================
    # DOWNLOAD RESULTS
    # ======================================================

    result = pd.DataFrame({
        "Date": y_test.index,
        "Actual Price": y_test.values,
        "Predicted Price": best_predictions
    })

    csv = result.to_csv(index=False)

    st.download_button(
        label="💾 Download Prediction Results",
        data=csv,
        file_name=f"{ticker}_prediction.csv",
        mime="text/csv"
    )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "Stock Price Prediction using Machine Learning | "
    "Educational Project"
)

st.caption(
    "Note: Predictions are uncertain and should not be treated "
    "as guaranteed future prices or financial advice."
)