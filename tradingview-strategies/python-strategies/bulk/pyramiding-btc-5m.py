#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "Pyramiding BTC 5 min"
timeframe = "5m"
leverage = 1

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def wma(series, length):
    weights = np.arange(1, length + 1)
    def calc_wma(x):
        if len(x) < length:
            return np.nan
        return np.dot(x, weights) / np.sum(weights)
    return series.rolling(window=length).apply(calc_wma, raw=True)

def hma(series, length):
    half = int(length / 2)
    sqrt_len = int(round(np.sqrt(length)))
    wma1 = wma(series, half)
    wma2 = wma(series, length)
    raw = 2 * wma1 - wma2
    return wma(raw, sqrt_len)

def linreg(series, length):
    values = np.full(len(series), np.nan)
    close_arr = series.to_numpy()
    for i in range(length - 1, len(series)):
        y = close_arr[i - length + 1 : i + 1]
        x = np.arange(length)
        coeffs = np.polyfit(x, y, 1)
        val = coeffs[0] * (length - 1) + coeffs[1]
        values[i] = val
    return pd.Series(values, index=series.index)

def crossover(series1, series2):
    s1 = series1.shift(1)
    s2 = series2.shift(1)
    return (series1 > series2) & (s1 <= s2)

def generate_signals(prices):
    if not isinstance(prices, pd.DataFrame):
        prices = pd.DataFrame(prices)
    
    df = prices.copy()
    n = len(df)
    signals = np.zeros(n, dtype=int)
    
    MAX_TRANCHES = 5
    PROFIT_PCT = 0.03
    LOSS_PCT = 0.10
    
    close = df['close']
    v1 = ema(close, 250)
    v2 = ema(close, 500)
    
    b = hma(close, 50).shift(1)
    linear_reg = linreg(close, 25)
    
    buy = crossover(linear_reg, b)
    trend_filter = (v1 > v2)
    
    # Strip Pine backtest-window gating in shared fair-comparison mode.
    long_signal = trend_filter & buy
    
    tranches = []
    
    for i in range(n):
        signals[i] = 1 if len(tranches) > 0 else 0
        
        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]
        
        surviving_tranches = []
        for t in tranches:
            hit_tp = current_high >= t['tp']
            hit_sl = current_low <= t['sl']
            if not (hit_tp or hit_sl):
                surviving_tranches.append(t)
        
        if long_signal.iloc[i] and len(surviving_tranches) < MAX_TRANCHES:
            entry_price = close.iloc[i]
            tp = entry_price * (1 + PROFIT_PCT)
            sl = entry_price * (1 - LOSS_PCT)
            surviving_tranches.append({'entry_price': entry_price, 'tp': tp, 'sl': sl})
        
        tranches = surviving_tranches
    
    return signals

if __name__ == "__main__":
    pass
