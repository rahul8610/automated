import model
import traceback
import sys

print("Starting debug 2")
try:
    ticker = 'RELIANCE.NS'
    fast_mode = False
    capital = 100000.0
    period = "5y"
    
    # Run the first half of fetch_and_train up to model creation to see NaNs
    import yfinance as yf
    stock = yf.Ticker(ticker)
    data_period = "1y" if fast_mode else "5y"
    df = stock.history(period=data_period)
    
    import ta
    import numpy as np
    
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

    # 2. Advanced Feature Engineering
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['EMA_20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['EMA_50'] = ta.trend.EMAIndicator(close=df['Close'], window=50).ema_indicator()
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
    df['Stoch_K'] = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close']).stoch()
    df['CCI'] = ta.trend.CCIIndicator(high=df['High'], low=df['Low'], close=df['Close']).cci()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    df['ROC'] = ta.momentum.ROCIndicator(close=df['Close'], window=12).roc()

    df['EMA_20_Dist'] = (df['Close'] - df['EMA_20']) / df['Close']
    df['EMA_50_Dist'] = (df['Close'] - df['EMA_50']) / df['Close']
    df['BB_High_Dist'] = (df['BB_High'] - df['Close']) / df['Close']
    df['BB_Low_Dist'] = (df['Close'] - df['BB_Low']) / df['Close']
    df['ATR_Pct'] = df['ATR'] / df['Close']
    df['Rolling_Std_Pct'] = df['Close'].rolling(window=20).std() / df['Close']
    
    df['Daily_Return'] = df['Close'].pct_change()
    df['Vol_Change'] = df['Volume'].pct_change()

    for i in range(1, 6):
        df[f'Return_Lag_{i}'] = df['Daily_Return'].shift(i)

    df['Target_Return'] = df['Close'].pct_change().shift(-1)
    df['Target_Direction'] = (df['Target_Return'] > 0).astype(int)
    
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
    print("df_clean length:", len(df_clean))

    features = [
        'Daily_Return', 'Vol_Change', 'RSI', 'MACD', 'MACD_Signal', 
        'EMA_20_Dist', 'EMA_50_Dist', 'BB_High_Dist', 'BB_Low_Dist', 
        'Rolling_Std_Pct', 'ATR_Pct', 'Stoch_K', 'CCI', 'ROC',
        'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3', 'Return_Lag_4', 'Return_Lag_5'
    ]
    
    X = df_clean[features]
    print("Is there any NaN in X?", X.isnull().sum())
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = X.iloc[:int(len(X)*0.8)]
    X_train_scaled = scaler.fit_transform(X_train)
    print("Is there any NaN in X_train_scaled?", np.isnan(X_train_scaled).sum())


except Exception as e:
    traceback.print_exc()

