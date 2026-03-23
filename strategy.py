def evaluate_strategies(current_price, predicted_price, current_rsi, current_macd, macd_signal, ema_short, ema_long, bb_lower, bb_upper, ml_prediction, ml_proba):
    strategies = {}
    
    # 1. Momentum Strategy
    if predicted_price > current_price and current_rsi < 70:
        strategies['Momentum'] = 'BUY'
    elif current_rsi > 70:
        strategies['Momentum'] = 'SELL'
    else:
        strategies['Momentum'] = 'HOLD'
        
    # 2. Trend Confirmation
    if ema_short > ema_long and current_macd > macd_signal:
        strategies['Trend'] = 'BUY'
    elif ema_short < ema_long and current_macd < macd_signal:
        strategies['Trend'] = 'SELL'
    else:
        strategies['Trend'] = 'HOLD'
        
    # 3. Mean Reversion
    if current_price < bb_lower:
        strategies['MeanReversion'] = 'BUY'
    elif current_price > bb_upper:
        strategies['MeanReversion'] = 'SELL'
    else:
        strategies['MeanReversion'] = 'HOLD'
        
    # 4. ML Confidence
    ml_conf = max(ml_proba) if ml_proba is not None and len(ml_proba) > 0 else 0 
    if ml_prediction == 1 and ml_conf > 0.55:
        strategies['ML'] = 'BUY'
    elif ml_prediction == 0 and ml_conf > 0.55:
        strategies['ML'] = 'SELL'
    else:
        strategies['ML'] = 'HOLD'
        
    # 5. Ensemble Voting
    votes = list(strategies.values())
    buy_votes = votes.count('BUY')
    sell_votes = votes.count('SELL')
    
    if buy_votes > sell_votes and buy_votes >= 2:
        final_signal = 'BUY'
    elif sell_votes > buy_votes and sell_votes >= 2:
        final_signal = 'SELL'
    else:
        final_signal = 'HOLD'
        
    # Generate Explanation
    reasons = []
    if strategies['Momentum'] != 'HOLD': reasons.append(f"Momentum: {strategies['Momentum']} (RSI: {current_rsi:.1f})")
    if strategies['Trend'] != 'HOLD': reasons.append(f"Trend: {strategies['Trend']} (EMA/MACD)")
    if strategies['MeanReversion'] != 'HOLD': reasons.append(f"Mean Reversion: {strategies['MeanReversion']}")
    if strategies['ML'] != 'HOLD': reasons.append(f"AI Model: {strategies['ML']} (Conf: {ml_conf*100:.1f}%)")
    
    explanation = " | ".join(reasons) if reasons else "Market showing mixed or neutral signals. Holdings advised."
    
    # Simple overall confidence calculation
    active_votes = buy_votes + sell_votes
    if active_votes == 0:
        overall_confidence = 50.0
    else:
        winning_votes = buy_votes if final_signal == 'BUY' else sell_votes if final_signal == 'SELL' else 0
        overall_confidence = (winning_votes / len(votes)) * 100
        
    # Boost by ML confidence if it agrees with the prevailing trend
    if strategies['ML'] == final_signal and final_signal != 'HOLD':
        overall_confidence = min(99.0, overall_confidence + (ml_conf * 15))
        
    return {
        "signal": final_signal,
        "confidence": round(overall_confidence, 2),
        "explanation": explanation,
        "strategy_breakdown": strategies
    }
