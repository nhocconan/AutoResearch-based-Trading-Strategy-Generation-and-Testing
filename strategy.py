#!/usr/bin/env python3
name = "6h_200EMA_Crossover_1dATR_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h EMA200 trend filter
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Daily ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_6h = align_htf_to_ltf(prices, df_1d, atr14)
    
    # Daily volatility threshold: 1.5x ATR14
    vol_threshold = atr14_6h * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if ATR not ready
        if np.isnan(atr14_6h[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: require sufficient volatility
        if high[i] - low[i] < vol_threshold[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above EMA200
            if close[i] > ema200[i] and close[i-1] <= ema200[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below EMA200
            elif close[i] < ema200[i] and close[i-1] >= ema200[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below EMA200
            if close[i] < ema200[i] and close[i-1] >= ema200[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above EMA200
            if close[i] > ema200[i] and close[i-1] <= ema200[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals