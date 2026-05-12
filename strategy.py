#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
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
    
    # Load 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align pivot levels to 1h timeframe (use previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + above 4h EMA50 + volume spike
            if close[i] > r1_aligned[i] and close[i] > ema_50_4h_aligned[i] and vol_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 + below 4h EMA50 + volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema_50_4h_aligned[i] and vol_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 or below 4h EMA50
            if close[i] < s1_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above R1 or above 4h EMA50
            if close[i] > r1_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals