from flask import Flask, render_template, request, jsonify
from model import fetch_and_train
from database import init_db, WatchlistItem, PredictionHistory, db
import os

app = Flask(__name__)

# Ensure templates and static directories exist
app.template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app.static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

# Initialize SQLite database on startup
init_db()

# --- Page Routes ---
@app.route('/')
def index():
    # We no longer handle POST here for the main app. It is heavily JS driven.
    return render_template('index.html')

@app.route('/watchlist')
def watchlist():
    db.connect(reuse_if_open=True)
    items = WatchlistItem.select().order_by(WatchlistItem.added_at.desc())
    watchlist_data = [{"ticker": item.ticker, "added": item.added_at.strftime('%Y-%m-%d')} for item in items]
    db.close()
    return render_template('watchlist.html', items=watchlist_data)

@app.route('/history')
def history():
    db.connect(reuse_if_open=True)
    records = PredictionHistory.select().order_by(PredictionHistory.date_run.desc()).limit(50)
    history_data = [
        {
            "ticker": r.ticker,
            "date": r.date_run.strftime('%Y-%m-%d %H:%M'),
            "current_price": r.current_price,
            "predicted_price": r.predicted_price,
            "suggestion": r.ai_suggestion
        } for r in records
    ]
    db.close()
    return render_template('history.html', history=history_data)

# --- API Routes (for JavaScript Async Fetch) ---
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    ticker = data.get('ticker', '').strip()
    
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400
        
    res, err = fetch_and_train(ticker)
    if err:
        return jsonify({"error": err}), 500
    
    return jsonify({"result": res})

@app.route('/api/watchlist', methods=['POST'])
def add_to_watchlist():
    data = request.get_json()
    ticker = data.get('ticker', '').strip()
    
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400
        
    try:
        db.connect(reuse_if_open=True)
        WatchlistItem.get_or_create(ticker=ticker.upper())
        db.close()
        return jsonify({"success": True, "message": f"{ticker.upper()} added to Watchlist!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
