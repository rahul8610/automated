Ultimate PredictPro Upgrade Plan
This plan details the implementation of all four major features to transform the script into a professional, high-performance web application.

1. Database Integration (Watchlist & History)
We will introduce peewee (a lightweight ORM) and a local SQLite database (predictpro.db).

[NEW] 
database.py
Define our schema:

WatchlistItem: Stores tickers the user has favorited.
PredictionHistory: Stores every prediction made (Ticker, Date, Current Price, Predicted Price, Suggestion) to track accuracy over time.
[MODIFY] 
app.py
Add routes for /watchlist and /history that query the database and render new templates. Add API routes to POST a ticker to the Watchlist.

[NEW] 
templates/watchlist.html
 & 
templates/history.html
Clean UI pages to display the database records in tabular or grid formats matching our premium aesthetic.

2. Model Persistence (Speed Optimization)
[MODIFY] 
model.py
Instead of training XGBoost for 5 seconds on every request, we will use joblib to save the trained .pkl model to a new /models/ directory natively.

Logic: If models/AAPL_xgb.pkl exists and is less than 24 hours old, load it instantly. If not, train it and save it.
3. Asynchronous Rendering (SPA Feel)
[MODIFY] 
templates/index.html
Transition the main page from generating via a full server POST to a smooth JavaScript 
fetch()
:

The user clicks "Explore".
JavaScript shows the loading spinner without reloading the page.
JavaScript pings a new /api/predict endpoint in 
app.py
.
When the data returns, JavaScript softly animates the glass cards into view and injects the dynamic values.
4. Interactive Charting
[MODIFY] 
templates/index.html
We will embed Chart.js via CDN. We will update 
model.py
 to return the last 30 days of actual historical prices alongside the prediction. JavaScript will render a beautiful, interactive line chart inside one of the main floating widget cards showing the stock's recent trajectory.

Dependency Updates
[MODIFY] 
requirements.txt
Add peewee and joblib.

Verification Plan
Re-install requirements and restart 
app.py
.
Generate a prediction for TSLA and verify the chart renders seamlessly without a page reload.
Verify that generating a second prediction for TSLA is nearly instantaneous (proving model caching worked).
Add TSLA to the Watchlist and verify it appears on the /watchlist tab.
