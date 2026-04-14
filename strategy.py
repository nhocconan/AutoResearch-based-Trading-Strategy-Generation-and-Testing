#!/usr/bin/env python3
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
    
    # Load 1d data for price channel and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR on 1d data
    tr = np.zeros_like(high_1d)
    for i in range(1, len(high_1d)):
        tr[i] = max(high_1d[i] - low_1d[i],
                    abs(high_1d[i] - high_1d[i-1]),
                    abs(low_1d[i] - low_1d[i-1]))
    
    atr_1d = np.full_like(high_1d, np.nan)
    if len(high_1d) >= 14:
        atr_1d[13] = np.mean(tr[1:14])
        for i in range(14, len(high_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate 1d high/low channels (20-period highest/lowest)
    highest_1d = np.full_like(high_1d, np.nan)
    lowest_1d = np.full_like(low_1d, np.nan)
    for i in range(19, len(high_1d)):
        highest_1d[i] = np.max(high_1d[i-19:i+1])
        lowest_1d[i] = np.min(low_1d[i-19:i+1])
    
    # Align 1d data to 12h timeframe
    highest_1d_aligned = align_htf_to_ltf(prices, df_1d, highest_1d)
    lowest_1d_aligned = align_htf_to_ltf(prices, df_1d, lowest_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate ATR on 12h for position sizing and stops
    tr_12h = np.zeros_like(high)
    for i in range(1, len(high)):
        tr_12h[i] = max(high[i] - low[i],
                        abs(high[i] - high[i-1]),
                        abs(low[i] - low[i-1]))
    
    atr_12h = np.full_like(high, np.nan)
    if len(high) >= 14:
        atr_12h[13] = np.mean(tr_12h[1:14])
        for i in range(14, len(high)):
            atr_12h[i] = (atr_12h[i-1] * 13 + tr_12h[i]) / 14
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_1d_aligned[i]) or np.isnan(lowest_1d_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: break above 20-period 1d high with volatility expansion
            if close[i] > highest_1d_aligned[i] and atr_12h[i] > atr_1d_aligned[i] * 1.5:
                position = 1
                signals[i] = position_size
            # Short entry: break below 20-period 1d low with volatility expansion
            elif close[i] < lowest_1d_aligned[i] and atr_12h[i] > atr_1d_aligned[i] * 1.5:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 10-period 1d low or volatility contraction
            if close[i] < lowest_1d_aligned[i] or atr_12h[i] < atr_1d_aligned[i] * 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 10-period 1d high or volatility contraction
            if close[i] > highest_1d_aligned[i] or atr_12h[i] < atr_1d_aligned[i] * 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Volatility_Channel_Breakout"
timeframe = "12h"
leverage = 1.0