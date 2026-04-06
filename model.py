import yfinance as yf
import pandas as pd
import xgboost as xgb
import ta
from datetime import datetime
import warnings
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import numpy as np
from database import PredictionHistory, ModelPerformance, db
from strategy import evaluate_strategies
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

warnings.simplefilter(action='ignore', category=FutureWarning)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

REGRESSORS = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.001), # Lowered alpha to prevent zeroing out
    'Decision Tree': DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_split=5, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, objective='reg:squarederror', random_state=42),
    'SVR': SVR(C=1.0, epsilon=0.01, kernel='rbf'),
    'KNN': KNeighborsRegressor(n_neighbors=7, weights='distance'),
    'AdaBoost': AdaBoostRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
}

clf1 = LogisticRegression(max_iter=2000, class_weight='balanced', C=0.05)
clf2 = RidgeClassifier(class_weight='balanced', alpha=10.0)
clf3 = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=15, class_weight='balanced', random_state=42)
clf4 = xgb.XGBClassifier(n_estimators=100, learning_rate=0.01, max_depth=2, subsample=0.5, colsample_bytree=0.5, eval_metric='logloss', random_state=42)
clf5 = HistGradientBoostingClassifier(learning_rate=0.01, max_depth=3, min_samples_leaf=20, l2_regularization=2.0, random_state=42)
clf6 = GaussianNB()

voting_clf = VotingClassifier(
    estimators=[
        ('lr', clf1), 
        ('ridge', clf2), 
        ('rf', clf3), 
        ('xgb', clf4),
        ('hgb', clf5)
    ],
    voting='hard'
)

CLASSIFIERS = {
    'Logistic Regression': clf1,
    'Ridge Classifier': clf2,
    'Random Forest': clf3,
    'XGBoost': clf4,
    'Hist Gradient Boosting': clf5,
    'Naive Bayes': clf6,
    'Expert Voting Ensemble': voting_clf
}

def fetch_and_train(ticker, fast_mode=False, capital=100000.0, period="5y"):
    """
    Fetches stock data, applies advanced stationary feature engineering,
    trains models to predict percentage returns, automatically selects the best, 
    saves it, and returns all UI/Chart data.
    """
    ticker = ticker.upper()
    try:
        scaler_path = os.path.join(MODELS_DIR, f"{ticker}_{period}_scaler.pkl")
        scaler_clf_path = os.path.join(MODELS_DIR, f"{ticker}_{period}_scaler_clf.pkl")
        best_regressor_path = os.path.join(MODELS_DIR, f"{ticker}_{period}_best_regressor.pkl")
        best_classifier_path = os.path.join(MODELS_DIR, f"{ticker}_{period}_best_classifier.pkl")
        clf_metrics_path = os.path.join(MODELS_DIR, f"{ticker}_{period}_clf_metrics.pkl")
        
        # 1. Fetch data
        stock = yf.Ticker(ticker)
        data_period = "1y" if fast_mode else "5y"
        df = stock.history(period=data_period) 
        
        # Indian Stock Auto-Correction fallback
        if df.empty and not ticker.endswith('.NS') and not ticker.endswith('.BO'):
            # Try appending .NS (National Stock Exchange of India)
            stock_in = yf.Ticker(ticker + ".NS")
            df_in = stock_in.history(period=data_period)
            if not df_in.empty:
                ticker = ticker + ".NS"
                stock = stock_in
                df = df_in
        
        if df.empty:
            return None, f"No data found for ticker '{ticker}'. For Indian stocks, trying appending .NS or .BO"
            
        try:
            indexes = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX']
            macro_data = yf.download(indexes, period=data_period, progress=False)['Close']
            
            safe_df_index = df.index.tz_convert(None) if getattr(df.index, 'tz', None) is not None else df.index
            safe_macro_index = macro_data.index.tz_convert(None) if getattr(macro_data.index, 'tz', None) is not None else macro_data.index
            macro_data.index = safe_macro_index
            
            macro_data = macro_data.reindex(safe_df_index, method='ffill')
            
            for idx in indexes:
                idx_name = idx.replace('^', '')
                df[f'{idx_name}_Close'] = macro_data[idx].values
                df[f'{idx_name}_Return'] = df[f'{idx_name}_Close'].pct_change()
                ema_20 = ta.trend.EMAIndicator(close=df[f'{idx_name}_Close'], window=20).ema_indicator()
                df[f'{idx_name}_EMA_20_Dist'] = (df[f'{idx_name}_Close'] - ema_20) / df[f'{idx_name}_Close']
                
            df['Day_of_Week'] = df.index.dayofweek
            df['Month'] = df.index.month
        except Exception as e:
            print(f"Warning: Failed to fetch macro indexes: {e}")
            
        # 2. Advanced Feature Engineering
        # Calculate standard technical indicators
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        macd = ta.trend.MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['EMA_20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
        df['EMA_50'] = ta.trend.EMAIndicator(close=df['Close'], window=50).ema_indicator()
        bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        
        # Additional powerful momentum/volatility indicators
        df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
        df['Stoch_K'] = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close']).stoch()
        df['CCI'] = ta.trend.CCIIndicator(high=df['High'], low=df['Low'], close=df['Close']).cci()
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
        df['ROC'] = ta.momentum.ROCIndicator(close=df['Close'], window=12).roc()

        # Stationarize Features! (CRITICAL for tree-based models & SVR stability)
        # We transform absolute prices into relative distances from current price.
        df['EMA_20_Dist'] = (df['Close'] - df['EMA_20']) / df['Close']
        df['EMA_50_Dist'] = (df['Close'] - df['EMA_50']) / df['Close']
        df['BB_High_Dist'] = (df['BB_High'] - df['Close']) / df['Close']
        df['BB_Low_Dist'] = (df['Close'] - df['BB_Low']) / df['Close']
        df['ATR_Pct'] = df['ATR'] / df['Close']
        df['Rolling_Std_Pct'] = df['Close'].rolling(window=20).std() / df['Close']
        
        df['Daily_Return'] = df['Close'].pct_change()
        df['Vol_Change'] = df['Volume'].pct_change()

        # Lag features (past 5 days explicitly to map autoregression)
        for i in range(1, 6):
            df[f'Return_Lag_{i}'] = df['Daily_Return'].shift(i)

        # TARGET VARIABLES: Predict the percentage return instead of absolute difference.
        df['Target_Return'] = df['Close'].pct_change().shift(-1)
        # Sincere non-leaking classification: predict tomorrow's direction reliably
        df['Target_Direction'] = (df['Target_Return'] > 0).astype(int)
        
        # Drop NaN
        df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
        if len(df_clean) < 50:
            return None, f"Not enough historical data for '{ticker}' after calculating indicators."

        # The pure stationary feature set
        features = [
            'Daily_Return', 'Vol_Change', 'RSI', 'MACD', 'MACD_Signal', 
            'EMA_20_Dist', 'EMA_50_Dist', 'BB_High_Dist', 'BB_Low_Dist', 
            'Rolling_Std_Pct', 'ATR_Pct', 'Stoch_K', 'CCI', 'ROC',
            'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3', 'Return_Lag_4', 'Return_Lag_5'
        ]
        
        features_clf = features + ['Day_of_Week', 'Month']
        for idx in ['GSPC', 'DJI', 'IXIC', 'RUT', 'VIX']:
            if f'{idx}_Return' in df_clean.columns:
                features_clf.extend([f'{idx}_Return', f'{idx}_EMA_20_Dist'])
                
        # Ensure absolutely no NaNs remain in features (e.g. from missing live volume or partial macro data)
        df_clean[features] = df_clean[features].ffill().fillna(0)
        df_clean[features_clf] = df_clean[features_clf].ffill().fillna(0)
        df[features] = df[features].ffill().fillna(0)
        df[features_clf] = df[features_clf].ffill().fillna(0)
        
        X = df_clean[features]
        X_clf_df = df_clean[features_clf]
        X_close = df_clean['Close']  # Kept for real-world metric conversion
        y_reg = df_clean['Target_Return']
        y_clf = df_clean['Target_Direction']
        
        performance_metrics = []
        clf_performance_metrics = []
        best_model_name = ""
        best_r2 = -float('inf')
        
        # 3. Model Persistence (Cache loading)
        if os.path.exists(best_regressor_path) and os.path.exists(best_classifier_path) and os.path.exists(scaler_path) and os.path.exists(scaler_clf_path) and os.path.exists(clf_metrics_path):
            best_regressor = joblib.load(best_regressor_path)
            best_classifier = joblib.load(best_classifier_path)
            scaler = joblib.load(scaler_path)
            scaler_clf = joblib.load(scaler_clf_path)
            clf_performance_metrics = joblib.load(clf_metrics_path)
            latest_data = df.copy()
            
            # Fetch performance from DB
            db.connect(reuse_if_open=True)
            db_metrics = ModelPerformance.select().where(ModelPerformance.ticker == ticker).order_by(ModelPerformance.timestamp.desc()).limit(10)
            
            if len(list(db_metrics)) > 0:
                for m in db_metrics:
                    performance_metrics.append({
                        'name': m.model_name,
                        'r2': m.r2_score,
                        'rmse': m.rmse,
                        'mae': m.mae
                    })
                best_model_name = max(performance_metrics, key=lambda x: x['r2'])['name']
            db.close()
            
            if len(performance_metrics) == 0:
                best_model_name = None

        if not best_model_name:
            # Requires Full Retraining... 
            # Strict TimeSeries equivalent split for pure out-of-sample validity
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            X_clf_train, X_clf_test = X_clf_df.iloc[:split_idx], X_clf_df.iloc[split_idx:]
            X_test_close = X_close.iloc[split_idx:]
            
            y_train_reg, y_test_reg = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
            y_train_clf, y_test_clf = y_clf.iloc[:split_idx], y_clf.iloc[split_idx:]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            scaler_clf = StandardScaler()
            X_clf_train_scaled = scaler_clf.fit_transform(X_clf_train)
            X_clf_test_scaled = scaler_clf.transform(X_clf_test)
            
            best_regressor = None
            
            db.connect(reuse_if_open=True)
            ModelPerformance.delete().where(ModelPerformance.ticker == ticker).execute()
            
            print(f"\n--- Training {len(REGRESSORS)} Advanced Regression Models ---")
            for name, model in REGRESSORS.items():
                model.fit(X_train_scaled, y_train_reg)
                preds_return = model.predict(X_test_scaled)
                
                # Convert the predicted percentage return back to absolute predicted price
                # for realistic performance evaluation against the actual dollar test set
                y_test_abs = X_test_close.values * (1 + y_test_reg.values)
                y_pred_abs = X_test_close.values * (1 + preds_return)
                
                rmse = np.sqrt(mean_squared_error(y_test_abs, y_pred_abs))
                mae = mean_absolute_error(y_test_abs, y_pred_abs)
                r2 = r2_score(y_test_abs, y_pred_abs)
                
                performance_metrics.append({
                    'name': name, 'r2': round(float(r2), 4), 'rmse': round(float(rmse), 2), 'mae': round(float(mae), 2)
                })
                
                ModelPerformance.create(
                    ticker=ticker, model_name=name, r2_score=round(float(r2), 4), rmse=round(float(rmse), 2), mae=round(float(mae), 2)
                )
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_model_name = name
                    best_regressor = model
            
            db.close()
            print(f"-> Best Regression Model: {best_model_name} (R2: {best_r2:.4f})")
            
            # Train Classifiers (Classification Layer)
            print(f"\n--- Training {len(CLASSIFIERS)} Advanced Classification Models ---")
            best_clf_score = -float('inf')
            best_classifier = None
            best_classifier_name = ""
            clf_performance_metrics = []
            
            for name, clf in CLASSIFIERS.items():
                clf.fit(X_clf_train_scaled, y_train_clf)
                preds = clf.predict(X_clf_test_scaled)
                acc = accuracy_score(y_test_clf, preds)
                clf_performance_metrics.append({'name': name, 'accuracy': round(float(acc), 4)})
                if acc > best_clf_score:
                    best_clf_score = acc
                    best_classifier_name = name
                    best_classifier = clf
            
            print(f"-> Best Classification Model: {best_classifier_name} (Accuracy: {best_clf_score:.4f})")
            
            # Retrain best models and scaler on ALL data before saving to cache
            X_full_scaled = scaler.fit_transform(X)
            X_clf_full_scaled = scaler_clf.fit_transform(X_clf_df)
            best_regressor.fit(X_full_scaled, y_reg)
            best_classifier.fit(X_clf_full_scaled, y_clf)
            
            joblib.dump(scaler, scaler_path)
            joblib.dump(scaler_clf, scaler_clf_path)
            joblib.dump(best_regressor, best_regressor_path)
            joblib.dump(best_classifier, best_classifier_path)
            joblib.dump(clf_performance_metrics, clf_metrics_path)
            latest_data = df.copy()
            print(f"\nBest Models successfully trained and saved to cache!\n")

        # 4. Generate Final Prediction
        latest_data = latest_data.ffill().bfill()
        latest_X = latest_data[features].iloc[-1:]
        current_price = latest_data['Close'].iloc[-1]
        
        # Fetch Real-Time Accurate Live Price
        try:
            live_price = stock.fast_info.last_price
            if live_price and not np.isnan(live_price) and live_price > 0:
                current_price = live_price
        except:
            pass
            
        # Pro Trading Levels (Standard Pivot Points based on latest candle)
        last_high = latest_data['High'].iloc[-1]
        last_low = latest_data['Low'].iloc[-1]
        last_close = latest_data['Close'].iloc[-1]
        
        pivot = (last_high + last_low + last_close) / 3
        r1 = (2 * pivot) - last_low
        s1 = (2 * pivot) - last_high
        r2 = pivot + (last_high - last_low)
        s2 = pivot - (last_high - last_low)
        
        latest_X_clf = latest_data[features_clf].iloc[-1:]
        
        latest_X_scaled = scaler.transform(latest_X)
        latest_X_clf_scaled = scaler_clf.transform(latest_X_clf)
        
        predicted_return = best_regressor.predict(latest_X_scaled)[0]
        prediction = current_price * (1 + predicted_return)
        
        direction_pred = best_classifier.predict(latest_X_clf_scaled)[0] # 1 for UP, 0 for DOWN
        
        if hasattr(best_classifier, "predict_proba"):
            direction_proba = best_classifier.predict_proba(latest_X_clf_scaled)[0]
        else:
            direction_proba = [0.5, 0.5]
        
        current_rsi = latest_data['RSI'].iloc[-1]
        current_macd = latest_data['MACD'].iloc[-1]
        macd_signal = latest_data['MACD_Signal'].iloc[-1]
        ema_short = latest_data['EMA_20'].iloc[-1]
        ema_long = latest_data['EMA_50'].iloc[-1]
        bb_lower = latest_data['BB_Low'].iloc[-1]
        bb_upper = latest_data['BB_High'].iloc[-1]
        
        # Apply Advanced Multi-Technique Strategy Engine
        strategy_results = evaluate_strategies(
            current_price=current_price, 
            predicted_price=prediction, 
            current_rsi=current_rsi, 
            current_macd=current_macd, 
            macd_signal=macd_signal, 
            ema_short=ema_short, 
            ema_long=ema_long, 
            bb_lower=bb_lower, 
            bb_upper=bb_upper, 
            ml_prediction=direction_pred, 
            ml_proba=direction_proba
        )
        
        suggestion = strategy_results['signal']
        confidence = strategy_results['confidence']
        suggestion_reasons = [strategy_results['explanation']]

        currency_code = stock.info.get('currency', 'USD').upper()
        currency_symbols = {'USD': '$', 'INR': '₹', 'EUR': '€', 'GBP': '£', 'JPY': '¥', 'CAD': 'C$', 'AUD': 'A$'}
        currency_sym = currency_symbols.get(currency_code, currency_code + " ")

        last_30_days = latest_data.tail(30)
        chart_labels = last_30_days.index.strftime('%m-%d').tolist()
        chart_data = last_30_days['Close'].round(2).tolist()
        chart_ema = last_30_days['EMA_20'].round(2).fillna(0).tolist()
        chart_ub = last_30_days['BB_High'].round(2).fillna(0).tolist()
        chart_lb = last_30_days['BB_Low'].round(2).fillna(0).tolist()

        db.connect(reuse_if_open=True)
        PredictionHistory.create(
            ticker=ticker,
            current_price=round(current_price, 2),
            predicted_price=round(float(prediction), 2),
            currency=currency_sym,
            ai_suggestion=suggestion,
            confidence_score=confidence,
            strategy_explanation=strategy_results['explanation'],
            model_used=best_model_name
        )
        db.close()
        
        # Run Backtesting Engine
        backtest_results = None
        if not fast_mode:
            from backtesting import run_backtest
            backtest_results = run_backtest(
                df=df_clean,
                classifier=best_classifier,
                scaler_clf=scaler_clf,
                features_clf=features_clf,
                ticker=ticker,
                initial_capital=capital
            )
            
        return {
            "ticker": ticker,
            "company_name": stock.info.get('longName', ticker),
            "currency": currency_sym,
            "current_price": round(float(current_price), 2),
            "predicted_price": round(float(prediction), 2),
            "price_diff": round(float(prediction - current_price), 2),
            "price_diff_pct": round(float(((prediction - current_price) / current_price) * 100), 2),
            "rsi": round(float(current_rsi), 2),
            "macd": round(float(current_macd), 4),
            "pivot": round(float(pivot), 2),
            "r1": round(float(r1), 2),
            "r2": round(float(r2), 2),
            "s1": round(float(s1), 2),
            "s2": round(float(s2), 2),
            "suggestion": suggestion,
            "confidence": confidence,
            "reasons": suggestion_reasons,
            "strategy_breakdown": strategy_results['strategy_breakdown'],
            "backtest": backtest_results,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chart_labels": chart_labels,
            "chart_data": chart_data,
            "chart_ema": chart_ema,
            "chart_ub": chart_ub,
            "chart_lb": chart_lb,
            "best_model": best_model_name,
            "comparison_metrics": sorted(performance_metrics, key=lambda x: x['r2'], reverse=True),
            "clf_comparison_metrics": sorted(clf_performance_metrics, key=lambda x: x['accuracy'], reverse=True)
        }, None
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"An error occurred while analyzing '{ticker}': {str(e)}"

if __name__ == '__main__':
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'NVDA'
    
    # We clear cache to force a full re-train test
    for ext in ['_best_regressor.pkl', '_best_classifier.pkl', '_scaler.pkl', '_scaler_clf.pkl', '_clf_metrics.pkl']:
        p = os.path.join(MODELS_DIR, f"{ticker}_5y{ext}")
        if os.path.exists(p): os.remove(p)
        
    result, error = fetch_and_train(ticker)
    if error:
        print(f"ERROR: {error}")
    else:
        print("\n" + "="*65)
        print(f"{'*** PREDICTPRO MODEL LEADERBOARD ***':^65}")
        print("="*65)
        print(f"BEST OVERALL MODEL: {result['best_model']}")
        print("-" * 65)
        print(f"{'Rank':<5} | {'Model Name':<22} | {'R² Score':<9} | {'RMSE':<7} | {'MAE':<7}")
        print("-" * 65)
        for i, m in enumerate(result['comparison_metrics'], 1):
            medal = "*" if i == 1 else " "
            print(f"{i:<2} {medal} | {m['name']:<22} | {m['r2']:<9.4f} | {m['rmse']:<7.2f} | {m['mae']:<7.2f}")
        print("="*65)
        
        print("\n" + "="*65)
        print(f"{'*** CLASSIFICATION MODEL LEADERBOARD ***':^65}")
        print("="*65)
        if result.get('clf_comparison_metrics'):
            print(f"BEST CLASSIFIER: {result['clf_comparison_metrics'][0]['name']}")
            print("-" * 65)
            print(f"{'Rank':<5} | {'Model Name':<30} | {'Accuracy':<9}")
            print("-" * 65)
            for i, m in enumerate(result['clf_comparison_metrics'], 1):
                medal = "*" if i == 1 else " "
                print(f"{i:<2} {medal} | {m['name']:<30} | {m['accuracy']:<9.4f}")
        print("="*65 + "\n")

