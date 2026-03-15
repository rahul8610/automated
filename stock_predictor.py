import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template_string, request
from datetime import datetime

app = Flask(__name__)

def fetch_and_train(ticker):
    """
    Fetches stock data, trains a lightweight model, and returns a prediction.
    """
    try:
        # 1. Fetch real-time/historical data automatically
        stock = yf.Ticker(ticker)
        df = stock.history(period="6mo") # 6 months is plenty for a lightweight model 
        
        if df.empty:
            return None, f"No data found for ticker '{ticker}'."
        
        # 2. Feature engineering: Use the closing prices of the past 3 days as features
        df['Prev_Close_1'] = df['Close'].shift(1)
        df['Prev_Close_2'] = df['Close'].shift(2)
        df['Prev_Close_3'] = df['Close'].shift(3)
        
        df.dropna(inplace=True)
        
        if len(df) < 10:
            return None, "Not enough data to train the model."

        X = df[['Prev_Close_1', 'Prev_Close_2', 'Prev_Close_3']]
        y = df['Close']
        
        # 3. Train lightweight machine learning model
        # Linear Regression is optimal for resource-constrained systems (negligible CPU/RAM usage)
        model = LinearRegression()
        model.fit(X, y)
        
        # 4. Predict the next close price using the most recent data
        latest_data = pd.DataFrame({
            'Prev_Close_1': [df['Close'].iloc[-1]],
            'Prev_Close_2': [df['Prev_Close_1'].iloc[-1]],
            'Prev_Close_3': [df['Prev_Close_2'].iloc[-1]]
        })
        
        prediction = model.predict(latest_data)[0]
        current_price = df['Close'].iloc[-1]
        
        # 5. Generate a basic trade suggestion based on predicted movement
        # If the model expects >0.5% growth, buy. If it expects >0.5% drop, sell.
        if prediction > current_price * 1.005: 
            suggestion = "BUY"
        elif prediction < current_price * 0.995: 
            suggestion = "SELL"
        else:
            suggestion = "HOLD"
        
        return {
            "ticker": ticker.upper(),
            "current_price": round(current_price, 2),
            "predicted_price": round(prediction, 2),
            "suggestion": suggestion,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, None
        
    except Exception as e:
        return None, f"An error occurred while processing '{ticker}': {str(e)}"

# Simple embedded front-end interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Stock Predictor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 500px; margin: 40px auto; padding: 20px; background: #f4f7f6; color: #333; }
        .card { background: white; padding: 25px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.05); }
        h2 { margin-top: 0; color: #2c3e50; text-align: center; }
        .form-group { display: flex; gap: 10px; margin-bottom: 20px; }
        input[type=text] { flex-grow: 1; padding: 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 16px; }
        button { padding: 12px 20px; background: #3498db; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: bold; transition: background 0.3s; }
        button:hover { background: #2980b9; }
        .result-box { margin-top: 20px; padding: 20px; border-radius: 8px; background: #f8f9fa; border: 1px solid #e9ecef; }
        .prediction-row { display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 16px; }
        .suggestion-box { text-align: center; margin-top: 15px; padding: 10px; border-radius: 6px; font-weight: bold; font-size: 18px; }
        .buy { background: #d4edda; color: #155724; border: 1px solid #c3e6cb;}
        .sell { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;}
        .hold { background: #e2e3e5; color: #383d41; border: 1px solid #d6d8db;}
        .footer { text-align: center; margin-top: 15px; font-size: 12px; color: #888; }
        .error { color: #721c24; background: #f8d7da; padding: 10px; border-radius: 6px; margin-top: 15px; text-align: center; }
    </style>
</head>
<body>
    <div class="card">
        <h2>AI Stock Predictor</h2>
        <form method="POST">
            <div class="form-group">
                <input type="text" name="ticker" placeholder="Enter Stock Ticker (e.g. AAPL, TSLA)" required autocomplete="off">
                <button type="submit">Predict</button>
            </div>
        </form>
        
        {% if result %}
            <div class="result-box">
                <h3 style="margin-top: 0; text-align: center; border-bottom: 1px solid #eee; padding-bottom: 10px;">{{ result.ticker }} Analysis</h3>
                <div class="prediction-row">
                    <span>Current Price:</span>
                    <strong>${{ result.current_price }}</strong>
                </div>
                <div class="prediction-row">
                    <span>Predicted Close:</span>
                    <strong>${{ result.predicted_price }}</strong>
                </div>
                <div class="suggestion-box {{ result.suggestion | lower }}">
                    TRADE SUGGESTION: {{ result.suggestion }}
                </div>
                <div class="footer">Updated: {{ result.date }}</div>
            </div>
        {% elif error %}
            <div class="error">
                {{ error }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    if request.method == 'POST':
        ticker = request.form.get('ticker', '').strip()
        if ticker:
            # Trigger our AI prediction workflow
            res, err = fetch_and_train(ticker)
            if err:
                error = err
            else:
                result = res
    return render_template_string(HTML_TEMPLATE, result=result, error=error)

if __name__ == '__main__':
    # Run a lightweight local development server
    app.run(host='0.0.0.0', port=5000, debug=True)
