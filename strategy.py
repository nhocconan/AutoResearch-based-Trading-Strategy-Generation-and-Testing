#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # Load daily data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Camarilla levels from previous day
    close_prev = np.roll(close_1d, 1)
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev[0] = np.nan
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    range_prev = high_prev - low_prev
    
    # Camarilla levels
    R4 = close_prev + range_prev * 1.500
    R3 = close_prev + range_prev * 1.250
    R2 = close_prev + range_prev * 1.166
    R1 = close_prev + range_prev * 1.083
    S1 = close_prev - range_prev * 1.083
    S2 = close_prev - range_prev * 1.166
    S3 = close_prev - range_prev * 1.250
    S4 = close_prev - range_prev * 1.500
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above R1 + above daily EMA34 + volume filter
            if high[i] > R1_aligned[i] and close[i] > ema_34_1d_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S1 + below daily EMA34 + volume filter
            elif low[i] < S1_aligned[i] and close[i] < ema_34_1d_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: breakdown below S1 or below daily EMA34
            if low[i] < S1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: breakout above R1 or above daily EMA34
            if high[i] > R1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals