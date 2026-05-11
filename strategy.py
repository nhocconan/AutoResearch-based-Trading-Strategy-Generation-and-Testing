#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_12h = close_12h > ema50_12h
    trend_up_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_up_12h)
    
    # Get 1d data for Camarilla levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day: based on previous day's OHLC
    # R1 = C + 1.1*(H-L)/12, S1 = C - 1.1*(H-L)/12
    # We need to shift by 1 day to avoid look-ahead (use previous day's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # First day has no previous day, set to 0 (will be handled by NaN check)
    prev_high_1d[0] = 0
    prev_low_1d[0] = 0
    prev_close_1d[0] = 0
    
    # Calculate Camarilla levels
    camarilla_R1 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 12
    camarilla_S1 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 12
    
    # Align to 4h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume moving average (20-period) for confirmation
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is invalid (NaN or zero from initialization)
        if (np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or
            np.isnan(trend_up_12h_aligned[i]) or
            np.isnan(vol_ma20[i]) or
            camarilla_R1_aligned[i] == 0 or
            camarilla_S1_aligned[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above Camarilla R1 + uptrend + volume confirmation
            if (close[i] > camarilla_R1_aligned[i] and 
                trend_up_12h_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below Camarilla S1 + downtrend + volume confirmation
            elif (close[i] < camarilla_S1_aligned[i] and 
                  not trend_up_12h_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Camarilla S1 or trend changes to down
            if (close[i] < camarilla_S1_aligned[i] or not trend_up_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Camarilla R1 or trend changes to up
            if (close[i] > camarilla_R1_aligned[i] or trend_up_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals