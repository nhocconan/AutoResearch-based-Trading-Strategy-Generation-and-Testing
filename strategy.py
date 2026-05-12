#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # Daily EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily range for Camarilla levels (previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_prev_1d = np.roll(close_1d, 1)
    close_prev_1d[0] = close_1d[0]  # first value
    
    range_1d = high_1d - low_1d
    # Camarilla levels
    R3 = close_prev_1d + range_1d * 1.1 / 4
    S3 = close_prev_1d - range_1d * 1.1 / 4
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R3 + above daily EMA34 + volume filter
            if high[i] > R3_aligned[i] and close[i] > ema_34_1d_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 + below daily EMA34 + volume filter
            elif low[i] < S3_aligned[i] and close[i] < ema_34_1d_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below S3 or below daily EMA34
            if low[i] < S3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above R3 or above daily EMA34
            if high[i] > R3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals