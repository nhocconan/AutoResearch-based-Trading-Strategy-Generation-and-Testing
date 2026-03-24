#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "Pyramiding BTC 5 min no security"
timeframe = "5m"
leverage = 1

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def wma(series, length):
    length = int(length)
    if length <= 0:
        return pd.Series(np.nan, index=series.index)
    weights = np.arange(1, length + 1)
    def calc(window):
        return np.dot(window, weights) / weights.sum()
    return series.rolling(window=length).apply(calc, raw=True)

def linreg(series, length, offset):
    length = int(length)
    def calc(window):
        x = np.arange(len(window))
        y = window
        if len(x) < 2:
            return np.nan
        slope, intercept = np.polyfit(x, y, 1)
        return slope * (len(window) - 1 + offset) + intercept
    return series.rolling(window=length).apply(calc, raw=True)

def hma3(series, length):
    p = int(length / 2)
    if p <= 0:
        return pd.Series(np.nan, index=series.index)
    p1 = max(1, int(p / 3))
    p2 = max(1, int(p / 2))
    p3 = max(1, p)
    w1 = wma(series, p1)
    w2 = wma(series, p2)
    w3 = wma(series, p3)
    src = w1 * 3 - w2 - w3
    return wma(src, p3)

def generate_signals(prices):
    df = prices.copy()
    n = len(df)
    signals = np.zeros(n, dtype=int)
    
    close = df['close']
    
    # Indicators
    hma_val = hma3(close, 50).shift(1)
    linreg_val = linreg(close, 25, 0)
    ema_fast = ema(close, 250)
    ema_slow = ema(close, 500)
    
    active_entries = []
    
    for i in range(n):
        # Signal reflects state entering bar i (1 if holding position)
        signals[i] = 1 if len(active_entries) > 0 else 0
        
        curr_close = close.iloc[i]
        curr_high = df['high'].iloc[i]
        curr_low = df['low'].iloc[i]
        
        # Check Exits (Dynamic targets based on current close)
        remaining_entries = []
        for entry_price in active_entries:
            profit_target = entry_price * 1.03
            loss_target = entry_price * 0.90
            
            hit_profit = curr_high >= profit_target
            hit_loss = curr_low <= loss_target
            
            if not (hit_profit or hit_loss):
                remaining_entries.append(entry_price)
        active_entries = remaining_entries
        
        # Check Entry (next bar execution - no same-bar fills)
        if i > 0:
            filter_ok = (ema_fast.iloc[i] > ema_slow.iloc[i])
            
            lr_curr = linreg_val.iloc[i]
            lr_prev = linreg_val.iloc[i-1]
            b_curr = hma_val.iloc[i]
            b_prev = hma_val.iloc[i-1]
            
            cross = False
            if not np.isnan(lr_curr) and not np.isnan(b_curr) and \
               not np.isnan(lr_prev) and not np.isnan(b_prev):
                cross = (lr_curr > b_curr) and (lr_prev <= b_prev)
            
            long_signal = filter_ok and cross
            
            if long_signal and len(active_entries) < 7:
                active_entries.append(curr_close)
                
    return signals
