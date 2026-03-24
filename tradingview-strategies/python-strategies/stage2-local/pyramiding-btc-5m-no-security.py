#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "Pyramiding BTC 5 min no security"
timeframe = "5m"

def ema(series, length):
    """Calculate Exponential Moving Average."""
    return series.ewm(span=length, adjust=False).mean()

def wma(series, length):
    """Calculate Weighted Moving Average."""
    length = int(length)
    if length < 1:
        return series.copy()
    weights = np.arange(1, length + 1)
    def rolling_wma(x):
        if len(x) < length:
            return np.nan
        return np.dot(x, weights) / weights.sum()
    return series.rolling(window=length).apply(rolling_wma, raw=True)

def hma3(series, length):
    """Calculate HMA3 variant from Pine script."""
    p = length / 2
    if p < 1:
        return series.copy()
    p_third = p / 3
    p_half = p / 2
    if p_third < 1:
        p_third = 1
    if p_half < 1:
        p_half = 1
    wma_p_third = wma(series, p_third)
    wma_p_half = wma(series, p_half)
    wma_p = wma(series, p)
    result = 3 * wma_p_third - wma_p_half - wma_p
    return wma(result, p)

def linear_regression(series, length):
    """Calculate linear regression value at each point."""
    def linreg_value(x):
        if len(x) < 2:
            return np.nan
        n = len(x)
        x_vals = np.arange(n)
        x_mean = x_vals.mean()
        y_mean = x.mean()
        numerator = np.sum((x_vals - x_mean) * (x - y_mean))
        denominator = np.sum((x_vals - x_mean) ** 2)
        if denominator == 0:
            return y_mean
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        return slope * (n - 1) + intercept
    return series.rolling(window=length).apply(linreg_value, raw=True)

def generate_signals(prices):
    """Generate target position signals for pyramiding strategy."""
    n = len(prices)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    
    signals = np.zeros(n, dtype=np.float64)
    close = prices['close'].values.astype(np.float64)
    
    close_series = prices['close'].astype(np.float64)
    
    fast_length = 250
    slow_length = 500
    hma_length = 50
    linreg_length = 25
    profit_pct = 0.03
    loss_pct = 0.10
    max_pyramid = 7
    filter_enabled = True
    
    v1 = ema(close_series, fast_length)
    v2 = ema(close_series, slow_length)
    
    hma3_val = hma3(close_series, hma_length)
    linreg_val = linear_regression(close_series, linreg_length)
    
    v1_arr = v1.values
    v2_arr = v2.values
    hma3_arr = hma3_val.values
    linreg_arr = linreg_val.values
    
    buy_signal = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(linreg_arr[i]) and not np.isnan(hma3_arr[i-1]):
            if linreg_arr[i] > hma3_arr[i-1] and linreg_arr[i-1] <= hma3_arr[i-1]:
                buy_signal[i] = True
    
    open_trades = 0
    entry_prices = []
    active_positions = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if i == 0:
            signals[i] = 0.0
            continue
        
        filter_cond = (v1_arr[i] > v2_arr[i]) or (not filter_enabled)
        
        if np.isnan(hma3_arr[i]) or np.isnan(linreg_arr[i]):
            signals[i] = signals[i-1]
            continue
        
        for entry_idx, entry_price in enumerate(entry_prices):
            if entry_price <= 0:
                continue
            high_price = prices['high'].iloc[i]
            low_price = prices['low'].iloc[i]
            
            tp_level = entry_price * (1 + profit_pct)
            sl_level = entry_price * (1 - loss_pct)
            
            if high_price >= tp_level or low_price <= sl_level:
                entry_prices[entry_idx] = 0
                open_trades -= 1
        
        entry_prices = [p for p in entry_prices if p > 0]
        open_trades = len(entry_prices)
        
        if buy_signal[i] and filter_cond and open_trades < max_pyramid:
            entry_prices.append(close[i-1])
            open_trades += 1
        
        if open_trades > 0:
            signals[i] = 1.0
        else:
            signals[i] = 0.0
    
    return signals
