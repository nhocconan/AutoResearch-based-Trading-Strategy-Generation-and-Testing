#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku cloud breakout with daily trend filter
# Uses Ichimoku conversion/base lines (9,26) and cloud (26,52) from 6h data
# Daily EMA(50) from 1d timeframe filters trades: only long when price > daily EMA, short when price < daily EMA
# Reduces false signals by aligning with higher timeframe trend
# Target: 50-150 total trades over 4 years = 12-37/year

name = "6h_1d_ichimoku_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 / 51) + (ema_50_1d[i-1] * 49 / 51)
    
    # Align daily EMA to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku calculations (9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = np.full(n, np.nan)
    low_9 = np.full(n, np.nan)
    for i in range(n):
        if i >= 8:
            high_9[i] = np.max(high[i-8:i+1])
            low_9[i] = np.min(low[i-8:i+1])
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = np.full(n, np.nan)
    low_26 = np.full(n, np.nan)
    for i in range(n):
        if i >= 25:
            high_26[i] = np.max(high[i-25:i+1])
            low_26[i] = np.min(low[i-25:i+1])
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Conversion + Base) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = np.full(n, np.nan)
    low_52 = np.full(n, np.nan)
    for i in range(n):
        if i >= 51:
            high_52[i] = np.max(high[i-51:i+1])
            low_52[i] = np.min(low[i-51:i+1])
    senkou_b = (high_52 + low_52) / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any required data is invalid
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        if position == 1:  # Long position
            # Exit: price closes below cloud OR below daily EMA
            if close[i] < cloud_bottom or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above cloud OR above daily EMA
            if close[i] > cloud_top or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above cloud AND above daily EMA
            if close[i] > cloud_top and close[i] > ema_50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below cloud AND below daily EMA
            elif close[i] < cloud_bottom and close[i] < ema_50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals