#!/usr/bin/env python3
name = "1d_Weekly_Camarilla_R3_S3_Breakout"
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
    
    # Get weekly data for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels using previous week
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    range_ = prev_week_high - prev_week_low
    
    # Camarilla levels: R3, R4, S3, S4
    r3 = pivot + range_ * 1.1 / 4
    r4 = pivot + range_ * 1.1 / 2
    s3 = pivot - range_ * 1.1 / 4
    s4 = pivot - range_ * 1.1 / 2
    
    # Align weekly levels to daily
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Daily trend filter: EMA50 > EMA200
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_up = ema50 > ema200
    trend_down = ema50 < ema200
    
    # Volume filter: current volume > 1.8x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R4 in daily uptrend with volume surge
            if (close[i] > r4_aligned[i] and 
                trend_up[i] and 
                volume[i] > 1.8 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 in daily downtrend with volume surge
            elif (close[i] < s4_aligned[i] and 
                  trend_down[i] and 
                  volume[i] > 1.8 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below R3 or trend changes
            if (close[i] < r3_aligned[i] or not trend_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above S3 or trend changes
            if (close[i] > s3_aligned[i] or not trend_down[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals