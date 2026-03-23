# PredictPro Advanced System - Execution Guide

Welcome to the upgraded PredictPro AI system! Because the application now features multiple interconnected advanced modules (like a dynamic strategy engine, historical backtesting simulators, and realtime model cache-polling), the execution flow happens in specific stages. 

Here is exactly how to start, interact with, and understand the workflow of the entire system.

***

## 📂 Architecture Overview
Before executing, here is a quick map of what each core Python file is responsible for:
- **`database.py`**: Boots up the local SQLite storage (`predictpro.db`) creating tables dynamically.
- **`strategy.py`**: The "brain" behind the trading signals. Evaluates Momentum, Trend, Mean Reversion, and AI predictions to cast an Ensemble vote.
- **`backtesting.py`**: Simulates historical trading bar-by-bar to calculate Profit/Loss, Win Rates, and Max Drawdown over a 5-year timeline.
- **`model.py`**: The heavy-lifting ML pipeline. Fetches Yahoo logic, computes features, manages cross-validation, trains/ranks models, and saves the winning models to a local `/models/` cache.
- **`app.py`**: The central networking hub. Manages the Flask HTTP Server and exposes the frontend APIs.

***

## 🚀 Step-by-Step Execution Journey

### Step 1: Start the Core Server
**Command to execute:**
To start the application, open your terminal, navigate to your folder (`c:\RAHUL\autom\automated`), and run the main entry point:
```powershell
python app.py
```
**What happens behind the scenes:**
1. `app.py` instantly imports `database.py`.
2. The database initializes and creates `predictpro.db` natively on your machine if it doesn't exist yet. 
3. Flask binds to your network port `5000`.

**Expected Output:**
The terminal will pause and display:
`* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)`

---

### Step 2: Access the User Interface
**Action:** 
Keep the terminal running. Open any web browser (Chrome, Edge, etc.) and type `http://127.0.0.1:5000` in the URL bar.

**What happens:**
Your browser downloads `index.html` and the sleek `style.css` designs without any data loaded yet.

**Expected Output:**
A clean, minimalist search page asking you to "Enter Ticker (e.g. NVDA, AAPL)".

---

### Step 3: Trigger the Analysis Pipeline (The Heavy Lift)
**Action:** 
Type a stock ticker (e.g., `AAPL`, `NVDA`, or `RELIANCE` for an Indian stock—the system will safely append `.NS` natively!) and click **"Explore Now"**.

**What happens computationally (The Chain Reaction):**
1. **Frontend:** Blurs out and sends a request to the backend route `POST /api/predict`.
2. **Backend (`model.py`):** 
   - Reaches out to Yahoo and pulls exactly 5 years of daily market trading data.
   - Calculates deep stationary technical arrays (RSI, Bollinger Bands, Moving Averages).
   - Trains exactly 10 Regression and 7 Classification models, using strict, realistic `TimeSeriesSplit` cross-validation.
   - Selects the #1 most accurate models, and directly caches them into your `/models/` folder.
3. **Strategy Engine (`strategy.py`):** Takes the raw probability outputs from that best AI model, alongside the mathematical technical arrays, and processes its multi-tech Rule Engine to return a **BUY**, **SELL**, or **HOLD** signal with a custom confidence percentage.
4. **Backtesting (`backtesting.py`):** Starts a virtual trading sequence with a pretend $100,000 using that exact Strategy. Sweeps through 1,200+ historical days and calculates exactly how much money that strategy would have yielded natively.
5. **Database (`database.py`):** Logs the payload and backtest run to your permanent history tables.

**Expected Output (Wait ~3-5 Seconds):**
The UI dramatically morphs revealing the Advanced Interactive Dashboard:
- The **Interactive Chart.js** maps the real stock price overlaid visually against Bollinger Bands and an EMA trail.
- The **Backtesting Dashboard** populates showing Win Rate and Total P/L generated over those 5 simulated years.
- The **Strategy Engines** cast their detailed breakdown alongside the Momentum pulse indicator.

---

### Step 4: The Real-Time Fast Polling Mode
**Action:** 
Simply do nothing! Leave the browser window open.

**What happens:**
1. Exactly every 30 seconds, `index.html` silently executes a background fetch to a secondary route: `POST /api/live_data`.
2. **Backend (`app.py`):** It triggers `fetch_and_train(fast_mode=True)` inside `model.py`.
3. Because `fast_mode` is True, it completely skips the 5-year timeline, skips training models, and skips backtesting! It instantly grabs your **pre-cached** `.pkl` models via joblib, pulls a tiny fraction of data to process the immediate RSI/MACD shifts today, and maps a realtime prediction in less than `1.0 second`.

**Expected Output:**
The top left ticker widget in your UI briefly flashes green and resets. Your charts, predicted price, and Momentum pulse dynamically shift if new volatility hit the market natively!
