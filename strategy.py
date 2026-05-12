#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
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
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 1d data for Camarilla pivot and volume filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d Camarilla pivot (based on previous day's range)
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    # R1 = Pivot + (H - L) * 1.1 / 2
    r1_1d = pivot_1d + (high_1d - low_1d) * 1.1 / 2
    # S1 = Pivot - (H - L) * 1.1 / 2
    s1_1d = pivot_1d - (high_1d - low_1d) * 1.1 / 2
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1d volume filter: current volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_filter_1d = volume_1d > (1.5 * vol_avg_20)
    vol_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_filter_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_filter_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if outside session
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 + above 4h EMA50 + volume filter
            if high[i] > r1_1d_aligned[i] and close[i] > ema_50_4h_aligned[i] and vol_filter_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: break below S1 + below 4h EMA50 + volume filter
            elif low[i] < s1_1d_aligned[i] and close[i] < ema_50_4h_aligned[i] and vol_filter_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: break below S1 or below 4h EMA50
            if low[i] < s1_1d_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: break above R1 or above 4h EMA50
            if high[i] > r1_1d_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals