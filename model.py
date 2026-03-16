import yfinance as yf
import pandas as pd
import xgboost as xgb
import ta
from datetime import datetime
import warnings
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from database import PredictionHistory, db

# Suppress pandas FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Ensure models directory exists
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def fetch_and_train(ticker):
    """
    Fetches stock data, trains an XGBoost model (or loads a cached one),
    saves the prediction to history, and returns all UI/Chart data.
    """
    ticker = ticker.upper()
    try:
        model_path = os.path.join(MODELS_DIR, f"{ticker}_xgb.pkl")
        model = None
        
        # 1. Fetch data
        stock = yf.Ticker(ticker)
        df = stock.history(period="5y") 
        
        if df.empty:
            return None, f"No data found for ticker '{ticker}'."
            
        # 2. Add Technical Indicators
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        macd = ta.trend.MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['EMA_20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
        df['EMA_50'] = ta.trend.EMAIndicator(close=df['Close'], window=50).ema_indicator()
        bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        
        # Target Variable: Next Day's Close
        df['Target_Next_Close'] = df['Close'].shift(-1)
        
        df_clean = df.dropna()
        if len(df_clean) < 50:
            return None, f"Not enough historical data for '{ticker}'."

        features = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'EMA_20', 'EMA_50', 'BB_High', 'BB_Low']
        
        # 3. Model Persistence (Cache loading)
        if os.path.exists(model_path):
            # Instant loading if model already exists
            model = joblib.load(model_path)
            # Only need latest data to predict
            latest_data = df.copy()
        else:
            # Full training cost only incurred once per ticker
            X = df_clean[features]
            y = df_clean['Target_Next_Close']
            
            # Split data for training/testing to get metrics (80% train, 20% test)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, objective='reg:squarederror')
            model.fit(X_train, y_train)
            
            # Print performance metrics to terminal
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"\n[{ticker}] MODEL RETRAINED - PERFORMANCE METRICS (on 20% holdout test data):")
            print(f"[{ticker}] -> MSE (Mean Squared Error):      {mse:.4f}")
            print(f"[{ticker}] -> RMSE (Root Mean Squared Error): {rmse:.4f}")
            print(f"[{ticker}] -> MAE (Mean Absolute Error):    {mae:.4f}")
            print(f"[{ticker}] -> R-Squared (Accuracy Score):   {r2:.4f} ({(r2*100):.2f}%)\n")
            
            # Train again on ALL data before saving for production use
            model.fit(X, y)
            joblib.dump(model, model_path)
            latest_data = df.copy()
        
        # 4. Predict
        latest_X = latest_data[features].iloc[-1:]
        prediction = model.predict(latest_X)[0]
        
        # Values for UI
        current_price = latest_data['Close'].iloc[-1]
        current_rsi = latest_data['RSI'].iloc[-1]
        current_macd = latest_data['MACD'].iloc[-1]
        
        # Generate Directives
        suggestion_reasons = []
        if prediction > current_price * 1.005: 
            suggestion = "BUY"
            suggestion_reasons.append(f"AI Model expects a price increase to ${prediction:.2f}")
        elif prediction < current_price * 0.995: 
            suggestion = "SELL"
            suggestion_reasons.append(f"AI Model expects a price drop to ${prediction:.2f}")
        else:
            suggestion = "HOLD"
            suggestion_reasons.append("AI Model expects sideways (flat) movement")
            
        if current_rsi > 70: suggestion_reasons.append("RSI indicates stock is currently Overbought")
        elif current_rsi < 30: suggestion_reasons.append("RSI indicates stock is currently Oversold")
            
        if current_macd > 0: suggestion_reasons.append("MACD shows bullish momentum")
        else: suggestion_reasons.append("MACD shows bearish momentum")

        # Determine Currency Symbol from Yahoo Finance Info
        currency_code = stock.info.get('currency', 'USD').upper()
        # Map some common currency codes to symbols (defaults to raw code if not found)
        currency_symbols = {
            'USD': '$',
            'INR': '₹',
            'EUR': '€',
            'GBP': '£',
            'JPY': '¥',
            'CAD': 'C$',
            'AUD': 'A$'
        }
        currency_sym = currency_symbols.get(currency_code, currency_code + " ")

        # 5. Extract trailing 30 days for Chart.js
        last_30_days = latest_data.tail(30)
        chart_labels = last_30_days.index.strftime('%m-%d').tolist()
        chart_data = last_30_days['Close'].round(2).tolist()

        # 6. Save to Database History
        db.connect(reuse_if_open=True)
        PredictionHistory.create(
            ticker=ticker,
            current_price=round(current_price, 2),
            predicted_price=round(float(prediction), 2),
            currency=currency_sym,
            ai_suggestion=suggestion
        )
        db.close()
            
        return {
            "ticker": ticker,
            "company_name": stock.info.get('longName', ticker),
            "currency": currency_sym,
            "current_price": round(current_price, 2),
            "predicted_price": round(float(prediction), 2),
            "price_diff": round(float(prediction - current_price), 2),
            "price_diff_pct": round(float(((prediction - current_price) / current_price) * 100), 2),
            "rsi": round(float(current_rsi), 2),
            "macd": round(float(current_macd), 4),
            "suggestion": suggestion,
            "reasons": suggestion_reasons,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chart_labels": chart_labels,
            "chart_data": chart_data
        }, None
        
    except Exception as e:
        return None, f"An error occurred while analyzing '{ticker}': {str(e)}"
