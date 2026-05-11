#!/usr/bin/env python3
name = "1d_1w_Camarilla_R1_S1_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Get weekly data for Camarilla pivots (from previous weekly bar)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous weekly bar's range
    range_1w = high_1w - low_1w
    
    # Calculate Camarilla R1 and S1 levels
    camarilla_r1 = close_1w + (range_1w * 1.1 / 12)
    camarilla_s1 = close_1w - (range_1w * 1.1 / 12)
    
    # Align Camarilla levels to daily timeframe (using previous weekly bar's values)
    r1_1d = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    s1_1d = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND above weekly EMA50 (uptrend) AND volume surge
            if close[i] > r1_1d[i] and close[i] > ema_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S1 AND below weekly EMA50 (downtrend) AND volume surge
            elif close[i] < s1_1d[i] and close[i] < ema_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price falls below S1 OR below weekly EMA50 (trend change)
            if close[i] < s1_1d[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30  # maintain position
        elif position == -1:
            # Short exit: price rises above R1 OR above weekly EMA50 (trend change)
            if close[i] > r1_1d[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30  # maintain position
    
    return signals