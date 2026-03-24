#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "CMARSI Strategy"
timeframe = "15m"
leverage = 1

def calculate_rsi(close_series, length):
    delta = close_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_updown(close_values):
    n = len(close_values)
    ud = np.zeros(n)
    for i in range(1, n):
        if close_values[i] == close_values[i-1]:
            ud[i] = 0
        elif close_values[i] > close_values[i-1]:
            if ud[i-1] <= 0:
                ud[i] = 1
            else:
                ud[i] = ud[i-1] + 1
        else:
            if ud[i-1] >= 0:
                ud[i] = -1
            else:
                ud[i] = ud[i-1] - 1
    return ud

def calculate_percentrank(series, length):
    def pr(x):
        if len(x) < 2:
            return np.nan
        current = x[-1]
        past = x[:-1]
        count = np.sum(past < current)
        return (count / len(past)) * 100
    return series.rolling(window=length).apply(pr, raw=True)

def generate_signals(prices):
    close = prices['close']
    
    # RSI on Close
    rsi = calculate_rsi(close, 3)
    
    # UpDown Indicator
    updown_vals = calculate_updown(close.values)
    updown_series = pd.Series(updown_vals, index=close.index)
    
    # RSI on UpDown
    updown_rsi = calculate_rsi(updown_series, 2)
    
    # Percent Rank of ROC
    roc = close.pct_change() * 100
    percentrank = calculate_percentrank(roc, 100)
    
    # Connors RSI Variant
    crsi = (rsi + updown_rsi + percentrank) / 3.0
    
    # Moving Average
    ma = crsi.rolling(window=2).mean()
    
    # Signal Generation
    signals = np.zeros(len(prices))
    position = 0
    
    band0 = 40
    band1 = 70
    
    ma_vals = ma.values
    
    for i in range(1, len(prices)):
        if np.isnan(ma_vals[i]) or np.isnan(ma_vals[i-1]):
            signals[i] = 0
            continue
            
        # Entry: Crossover MA > 40
        if ma_vals[i] > band0 and ma_vals[i-1] <= band0:
            position = 1
        # Exit: Crossunder MA < 70
        elif ma_vals[i] < band1 and ma_vals[i-1] >= band1:
            position = 0
            
        signals[i] = position
        
    return signals
