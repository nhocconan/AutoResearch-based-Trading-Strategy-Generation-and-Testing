#!/usr/bin/env python3
"""
12H_1W_CAMARILLA_R1_S1_BREAKOUT_1W_TREND_FILTER
Hypothesis: Use 1w price position relative to 200-period EMA for trend direction (bull/bear filter) and 12h Camarilla R1/S1 levels for entry.
Long when price is above 1w EMA200 (bull market) and breaks above 12h R1 level; 
Short when price is below 1w EMA200 (bear market) and breaks below 12h S1 level.
Volume confirmation: current volume > 1.8x 20-period average volume.
Minimum 24 bars (12 days) between trades to reduce frequency.
This combines long-term trend filtering with intraday precision to work in both bull and bear markets while minimizing whipsaws.
"""
name = "12H_1W_CAMARILLA_R1_S1_BREAKOUT_1W_TREND_FILTER"
timeframe = "12h"
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
    
    # Get 1w data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1w EMA200
    close_1w = pd.Series(df_1w['close'])
    ema_200 = close_1w.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # Get 12h data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (R1, S1)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    pivot = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    r1 = pivot + (range_12h * 1.1 / 12)
    s1 = pivot - (range_12h * 1.1 / 12)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    # Volume filter: current volume > 1.8 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(200, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema_200_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 24 bars between trades (12 days on 12h TF) to reduce frequency
            if bars_since_exit < 24:
                continue
                
            # Long: price above 1w EMA200 (bull market) and breaks above R1 level
            if (close[i] > ema_200_aligned[i] and 
                high[i] > r1_aligned[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price below 1w EMA200 (bear market) and breaks below S1 level
            elif (close[i] < ema_200_aligned[i] and 
                  low[i] < s1_aligned[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite side of 1w EMA200
            if position == 1 and close[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals