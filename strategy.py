#!/usr/bin/env python3
name = "1d_1w_Camarilla_R1_S1_Breakout_Trend_Volume"
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
    
    # Get 1w data for trend filter (using 1w EMA50)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Get 1d data for Camarilla pivots (from previous 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous 1d bar's range
    range_1d = high_1d - low_1d
    
    # Calculate Camarilla R1 and S1 levels
    camarilla_r1 = close_1d + (range_1d * 1.1 / 12)
    camarilla_s1 = close_1d - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 1d timeframe (using previous 1d bar's values)
    r1_1d = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_1d = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
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
            # Long: price breaks above R1 AND above 1w EMA50 (uptrend) AND volume surge
            if close[i] > r1_1d[i] and close[i] > ema_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S1 AND below 1w EMA50 (downtrend) AND volume surge
            elif close[i] < s1_1d[i] and close[i] < ema_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price falls below S1 OR below 1w EMA50 (trend change)
            if close[i] < s1_1d[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30  # maintain position
        elif position == -1:
            # Short exit: price rises above R1 OR above 1w EMA50 (trend change)
            if close[i] > r1_1d[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30  # maintain position
    
    return signals