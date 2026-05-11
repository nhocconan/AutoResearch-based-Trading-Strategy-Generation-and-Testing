#!/usr/bin/env python3
name = "1h_4h1d_Camarilla_R1S1_Breakout_Volume"
timeframe = "1h"
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
    
    # Get 4H data for trend context
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4H EMA50 for trend direction
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1D data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels (R1, S1)
    # For current 1h bar, use previous day's high/low/close
    prev_high = df_1d['high'].shift(1).values  # Previous day high
    prev_low = df_1d['low'].shift(1).values    # Previous day low
    prev_close = df_1d['close'].shift(1).values # Previous day close
    
    # Camarilla R1 and S1
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: volume > 1.5x 24-period average
    vol_ma24 = np.zeros(n)
    for i in range(n):
        if i < 24:
            vol_ma24[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 4H EMA50
        uptrend = close[i] > ema50_4h_aligned[i]
        downtrend = close[i] < ema50_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 in uptrend with volume surge
            if (close[i] > r1_aligned[i] and uptrend and 
                volume[i] > 1.5 * vol_ma24[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 in downtrend with volume surge
            elif (close[i] < s1_aligned[i] and downtrend and 
                  volume[i] > 1.5 * vol_ma24[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 or trend changes
            if (close[i] < s1_aligned[i] or not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above R1 or trend changes
            if (close[i] > r1_aligned[i] or not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals