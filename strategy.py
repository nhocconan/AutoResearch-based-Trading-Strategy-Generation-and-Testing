#!/usr/bin/env python3
name = "1d_Camarilla_Pivot_WeeklyTrend"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Weekly EMA21 trend
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    trend_up_1w = close_1w > ema21_1w
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    
    # Daily Camarilla pivot calculation
    # Pivot = (H + L + C) / 3
    pivot = (high + low + close) / 3
    # R1 = C + (H - L) * 1.1/12
    # S1 = C - (H - L) * 1.1/12
    r1 = close + (high - low) * 1.1 / 12
    s1 = close - (high - low) * 1.1 / 12
    
    # Volume filter: current volume > 1.3x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.3 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(pivot[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or
            np.isnan(trend_up_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close crosses above R1 + weekly uptrend + volume confirmation
            if close[i] > r1[i] and close[i-1] <= r1[i-1] and trend_up_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close crosses below S1 + weekly downtrend + volume confirmation
            elif close[i] < s1[i] and close[i-1] >= s1[i-1] and not trend_up_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close crosses below pivot OR weekly trend turns down
            if close[i] < pivot[i] and close[i-1] >= pivot[i-1] or not trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close crosses above pivot OR weekly trend turns up
            if close[i] > pivot[i] and close[i-1] <= pivot[i-1] or trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals