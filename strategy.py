#!/usr/bin/env python3
name = "1d_WeeklyPivot_R1S1_Breakout_1wEMA_Trend_Volume"
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
    
    # Get weekly data for trend filter (EMA50) and pivot point calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate weekly pivot points (R1, S1) from previous week's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = (high_1w + low_1w + close_1w) / 3
    # R1 = (2 * PP) - L
    r1 = (2 * pp) - low_1w
    # S1 = (2 * PP) - H
    s1 = (2 * pp) - high_1w
    
    # Shift to get previous week's values (to avoid look-ahead)
    r1_shifted = np.roll(r1, 1)
    s1_shifted = np.roll(s1, 1)
    
    # Align pivot levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_shifted)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_shifted)
    
    # Calculate volume confirmation (current volume vs 20-day average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup for weekly calculations
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 level, weekly uptrend (price > EMA50), volume confirmation
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S1 level, weekly downtrend (price < EMA50), volume confirmation
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 level (reversal signal)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price breaks above R1 level (reversal signal)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals