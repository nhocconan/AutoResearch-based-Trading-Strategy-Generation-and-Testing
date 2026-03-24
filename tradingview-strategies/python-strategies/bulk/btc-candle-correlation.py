#!/usr/bin/env python3
import pandas as pd
import numpy as np

name = "BTC Candle Correlation Strategy"
timeframe = "4h"
leverage = 1

def generate_signals(prices):
    df = prices.copy()
    n = len(df)
    if n < 10:
        return np.zeros(n)
    
    h2 = df['high'] ** 2
    l2 = df['low'] ** 2
    o2 = df['open'] ** 2
    c2 = df['close'] ** 2
    source = np.sqrt((h2 + l2 + o2 + c2) / 8.0)
    
    length = 9
    mult = 0.9
    
    ma = source.rolling(window=length).mean()
    range_val = df['high'] - df['low']
    rangema = range_val.rolling(window=length).mean()
    
    upper = ma + rangema * mult
    lower = ma - rangema * mult
    
    src_vals = source.values
    up_vals = upper.values
    low_vals = lower.values
    close_vals = df['close'].values
    open_vals = df['open'].values
    
    cross_upper = np.zeros(n, dtype=bool)
    cross_lower = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        if src_vals[i] > up_vals[i] and src_vals[i-1] <= up_vals[i-1]:
            cross_upper[i] = True
        if src_vals[i] < low_vals[i] and src_vals[i-1] >= low_vals[i-1]:
            cross_lower[i] = True
            
    bullish_candle = close_vals > open_vals
    entry_signal = cross_upper & bullish_candle
    exit_signal = cross_lower
    
    positions = np.zeros(n)
    in_long = False
    
    for i in range(1, n):
        positions[i] = 1.0 if in_long else 0.0
        if entry_signal[i] and not in_long:
            in_long = True
        elif exit_signal[i] and in_long:
            in_long = False
            
    return positions
