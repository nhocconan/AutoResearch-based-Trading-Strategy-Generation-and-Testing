#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50 = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R4 = close + 1.5 * (high - low) * 1.1/2, R3 = close + 1.25 * (high - low) * 1.1/2
    #          R2 = close + 1.166 * (high - low) * 1.1/2, R1 = close + 1.083 * (high - low) * 1.1/2
    #          S1 = close - 1.083 * (high - low) * 1.1/2, S2 = close - 1.166 * (high - low) * 1.1/2
    #          S3 = close - 1.25 * (high - low) * 1.1/2, S4 = close - 1.5 * (high - low) * 1.1/2
    # We use R1 and S1 as entry levels
    range_1d = high_1d - low_1d
    r1_1d = close_1d + 1.083 * range_1d * 1.1 / 2
    s1_1d = close_1d - 1.083 * range_1d * 1.1 / 2
    
    # Align 1d R1/S1 to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 4h volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 50)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + uptrend (12h EMA50) + volume
            if close[i] > r1_1d_aligned[i] and close[i] > ema_50_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + downtrend (12h EMA50) + volume
            elif close[i] < s1_1d_aligned[i] and close[i] < ema_50_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price crosses back through the pivot level
            if position == 1:
                if close[i] < s1_1d_aligned[i]:  # Exit at S1 level
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > r1_1d_aligned[i]:  # Exit at R1 level
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals