import pandas as pd
import numpy as np
from strategy import evaluate_strategies
from database import BacktestSummary, db

def run_backtest(df, classifier, scaler_clf, features_clf, ticker, initial_capital=100000.0):
    """
    Simulate trading over the historical dataset using the ensemble strategy logic.
    Assumes `df` contains all pre-calculated indicators (EMA, MACD, RSI, BB, etc.).
    """
    
    # Generate historical ML predictions
    X_clf = df[features_clf]
    X_clf_scaled = scaler_clf.transform(X_clf)
    
    ml_predictions = classifier.predict(X_clf_scaled)
    if hasattr(classifier, "predict_proba"):
        ml_probas = classifier.predict_proba(X_clf_scaled)
    else:
        # Fallback pseudo-probabilities
        ml_probas = [[0.5, 0.5] if p == 0 else [0.2, 0.8] for p in ml_predictions]
        
    capital = initial_capital
    position = 0 # Number of shares held
    buy_price = 0.0
    
    trades = 0
    winning_trades = 0
    
    equity_curve = []
    peak_capital = capital
    max_drawdown = 0.0
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        current_price = row['Close']
        current_rsi = row['RSI']
        current_macd = row['MACD']
        macd_signal = row.get('MACD_Signal', 0)
        ema_short = row.get('EMA_20', 0)
        ema_long = row.get('EMA_50', 0)
        bb_lower = row.get('BB_Low', 0)
        bb_upper = row.get('BB_High', 0)
        
        # We need predicted price conceptually, but here we can just use the target or skip Momentum's price check
        # For backtesting, if we bought today, we pretend we're executing at the 'Close'
        predicted_price = current_price * (1 + row.get('Target_Return', 0))
        
        ml_pred = ml_predictions[i]
        ml_prob = ml_probas[i]
        
        strat = evaluate_strategies(
            current_price=current_price,
            predicted_price=predicted_price,
            current_rsi=current_rsi,
            current_macd=current_macd,
            macd_signal=macd_signal,
            ema_short=ema_short,
            ema_long=ema_long,
            bb_lower=bb_lower,
            bb_upper=bb_upper,
            ml_prediction=ml_pred,
            ml_proba=ml_prob
        )
        
        signal = strat['signal']
        
        # Execution Engine
        if signal == 'BUY' and position == 0:
            # Buy max shares
            position = capital / current_price
            buy_price = current_price
            capital = 0
        elif signal == 'SELL' and position > 0:
            # Sell all shares
            revenue = position * current_price
            capital += revenue
            
            if revenue > (position * buy_price):
                winning_trades += 1
            trades += 1
            position = 0
            buy_price = 0
            
        # Update Equity Curve & Drawdown
        current_equity = capital if position == 0 else (position * current_price)
        equity_curve.append({
            'date': df.index[i].strftime('%Y-%m-%d'),
            'equity': round(current_equity, 2)
        })
        
        if current_equity > peak_capital:
            peak_capital = current_equity
        
        drawdown = (peak_capital - current_equity) / peak_capital
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            
    # Close out any open positions at the end of the test
    if position > 0:
        final_equity = position * df.iloc[-1]['Close']
        capital += final_equity
        if final_equity > (position * buy_price):
            winning_trades += 1
        trades += 1
        position = 0
        
    total_pnl = capital - initial_capital
    win_rate = (winning_trades / trades * 100) if trades > 0 else 0.0
    roi = (total_pnl / initial_capital * 100) if initial_capital > 0 else 0.0
    
    # Save to Database
    db.connect(reuse_if_open=True)
    BacktestSummary.create(
        ticker=ticker,
        initial_capital=initial_capital,
        final_capital=round(capital, 2),
        total_pnl=round(total_pnl, 2),
        win_rate=round(win_rate, 2),
        total_trades=trades,
        max_drawdown=round(max_drawdown * 100, 2)
    )
    db.close()
    
    return {
        "initial_capital": initial_capital,
        "final_capital": round(capital, 2),
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(win_rate, 2),
        "roi": round(roi, 2),
        "total_trades": trades,
        "max_drawdown": round(max_drawdown * 100, 2),
        "equity_curve": equity_curve
    }
