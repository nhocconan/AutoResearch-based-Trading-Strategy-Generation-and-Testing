#!/usr/bin/env python3
name = "1d_WeeklyPivot_R1S1_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # Get weekly data for pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA30 for trend filter
    close_1w = df_1w['close'].values
    ema_30_1w = pd.Series(close_1w).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_30_1w)
    
    # Calculate weekly pivot points (R1, S1)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Shift to get previous week's values
    high_1w_shifted = np.roll(high_1w, 1)
    low_1w_shifted = np.roll(low_1w, 1)
    close_1w_shifted = np.roll(close_1w, 1)
    
    # Calculate pivot point and R1/S1
    pivot = (high_1w_shifted + low_1w_shifted + close_1w_shifted) / 3
    r1 = 2 * pivot - low_1w_shifted
    s1 = 2 * pivot - high_1w_shifted
    
    # Align pivot levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Calculate volume confirmation (current volume vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_30_1w_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 level, uptrend (price > EMA30), volume confirmation
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_30_1w_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 level, downtrend (price < EMA30), volume confirmation
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_30_1w_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 level (reversal signal)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 level (reversal signal)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals