#!/usr/bin/env python3
name = "6h_1d_Camarilla_R4S4_Breakout_Trend_Volume"
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
    
    # Get 1d data for Camarilla levels (R4, S4) and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R4, S4) from previous day
    R4 = np.zeros(len(high_1d))
    S4 = np.zeros(len(high_1d))
    
    for i in range(len(high_1d)):
        if i < 1:
            R4[i] = np.nan
            S4[i] = np.nan
        else:
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d[i-1]
            range_val = prev_high - prev_low
            R4[i] = prev_close + range_val * 1.1 / 2
            S4[i] = prev_close - range_val * 1.1 / 2
    
    # Trend filter: 20 EMA on 1d
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_up_1d = close_1d > ema20_1d
    
    # Align indicators to 6h timeframe
    R4_6h_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_6h_aligned = align_htf_to_ltf(prices, df_1d, S4)
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    # Volume moving average (20-period) for confirmation
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(R4_6h_aligned[i]) or 
            np.isnan(S4_6h_aligned[i]) or
            np.isnan(trend_up_1d_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R4 + uptrend + volume confirmation
            if (close[i] > R4_6h_aligned[i] and 
                trend_up_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 + downtrend + volume confirmation
            elif (close[i] < S4_6h_aligned[i] and 
                  not trend_up_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S4 or trend changes
            if (close[i] < S4_6h_aligned[i] or not trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R4 or trend changes
            if (close[i] > R4_6h_aligned[i] or trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals