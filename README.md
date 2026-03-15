# PredictPro - Market Intelligence. Perfected.

**PredictPro** is a modern, high-performance Web Application that uses advanced Machine Learning (XGBoost) and AI to predict short-term stock price movements based on 5 years of historical financial data and momentum indicators. 

Featuring an ultra-premium, Apple-inspired light-mode UI, it acts as a fully asynchronous Single Page Application (SPA), delivering instant interactive analytics.

## 🌟 Key Features

1. **AI-Powered "Quantum" Intelligence:**
   - Automatically downloads 5 years of historical stock data via the `yfinance` API.
   - Calculates advanced technical indicators on-the-fly (`RSI`, `MACD`, `Bollinger Bands`, `EMAs`).
   - Uses an **XGBoost Regressor** model to forecast the following day's closing price.
2. **Instant Model Caching (Joblib):**
   - The first prediction trains a unique XGBoost model for that ticker.
   - Subsequent predictions load the serialized `.pkl` model instantly, slashing wait times to milliseconds.
3. **Database Integration (SQLite + Peewee):**
   - **Watchlist:** Save your favorite tickers to easily track and analyze them with one click.
   - **History Tracking:** Automatically logs every AI prediction (including price, forecast, and date) to monitor AI accuracy over time.
4. **Interactive Dashboard:**
   - Asynchronous data fetching (`fetch()` API) guarantees the page never reloads.
   - Beautiful, interactive historical line graphs rendered natively with **Chart.js**.

---

## 🛠️ Tech Stack

- **Backend:** Python (Flask, Peewee ORM, Joblib)
- **AI & Data Science:** XGBoost, Scikit-Learn, Pandas, TA (Technical Analysis), Yfinance
- **Frontend UI:** HTML5, Premium custom CSS (Glassmorphism), Vanilla JavaScript
- **Data Visualization:** Chart.js
- **Database:** SQLite

---

## 🚀 Setup & Installation Instructions

This project uses `uv` as a blazing-fast Python package and project manager.

### 1. Prerequisites
Ensure you have Python 3 installed on your system.

### 2. Install Dependencies
Run the following pip/uv command in your terminal targeting the requirements file:

```bash
pip install -r requirements.txt
# OR if using uv
uv pip install -r requirements.txt
```

### 3. Run the Application
The entire application logic is contained within `app.py`, which integrates the web routes, asynchronous API endpoints, and the AI `model.py` core.

Ensure you are located in the `automated` directory, and run:

```bash
python app.py
# OR if running natively tied to uv
uv run app.py
```

### 4. Access the Dashboard
The Flask server will start locally. Open your favorite web browser and navigate to:

```text
http://localhost:5000
```

---

## 📂 Project Structure

- **`app.py`**: The Flask server. Handles all UI routing and asynchronous JSON endpoints (`/api/predict`).
- **`model.py`**: The core AI logic. Fetches Yahoo Finance data, generates technical features, handles XGBoost model training/caching, and formulates trading suggestions.
- **`database.py`**: Sets up the SQLite database and Peewee ORM structures for the Watchlist and History tracking.
- **`templates/`**: Contains the HTML structures.
  - `index.html`: The main analytical dashboard and SPA core.
  - `watchlist.html`: The saved stock tracking interface.
  - `history.html`: The log of all past AI actions.
- **`static/style.css`**: The premium light-theme stylesheet powering the modern "Apple-esque" aesthetic.
- **`models/`**: (Auto-generated) Stores the cached `.pkl` XGBoost models for instant loading.
- **`predictpro.db`**: (Auto-generated) The SQLite database file.