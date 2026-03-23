from model import fetch_and_train

print("Testing Reliance without .NS suffix...")
res, err = fetch_and_train("RELIANCE")
if err:
    print("Error:", err)
else:
    print("Success! Return object keys:", res.keys())
    print("Backtest PNL:", res['backtest']['total_pnl'])
    print("Win Rate:", res['backtest']['win_rate'], "%")
    print("Ensemble Signal:", res['suggestion'], "(Conf:", res['confidence'], "%)")
    print("Strategy Breakdown:", res['strategy_breakdown'])
