#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h 123 Reversal pattern with weekly pivot confirmation
# Long when 123 Reversal bullish pattern forms AND price above weekly pivot (bullish bias)
# Short when 123 Reversal bearish pattern forms AND price below weekly pivot (bearish bias)
# 123 Reversal: Point 1 (swing low/high), Point 2 (retracement), Point 3 (failure to make new low/high)
# Weekly pivot provides higher timeframe bias to filter counter-trend signals
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === Weekly Pivot (higher timeframe bias) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pp_1w - low_1w
    s1_1w = 2 * pp_1w - high_1w
    
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # === 123 Reversal Pattern Detection ===
    # Point 1: Swing point (using 5-bar window)
    # Point 2: Retracement from Point 1
    # Point 3: Failed attempt to exceed Point 1
    
    # Find swing highs and lows
    window = 5
    swing_high = np.zeros(n)
    swing_low = np.zeros(n)
    
    for i in range(window, n - window):
        # Swing high: highest high in window
        if high[i] == np.max(high[i-window:i+window+1]):
            swing_high[i] = high[i]
        # Swing low: lowest low in window
        if low[i] == np.min(low[i-window:i+window+1]):
            swing_low[i] = low[i]
    
    # Initialize pattern tracking arrays
    bull_123 = np.zeros(n, dtype=bool)  # Bullish 123 reversal
    bear_123 = np.zeros(n, dtype=bool)  # Bearish 123 reversal
    
    # Track last swing points
    last_swing_high = 0
    last_swing_low = 0
    last_swing_high_idx = -1
    last_swing_low_idx = -1
    
    for i in range(window, n):
        # Update swing points
        if swing_high[i] > 0:
            last_swing_high = swing_high[i]
            last_swing_high_idx = i
        if swing_low[i] > 0:
            last_swing_low = swing_low[i]
            last_swing_low_idx = i
        
        # Bullish 123: After swing low, price rallies (Point 2), then fails to make new low (Point 3)
        if (last_swing_low_idx > 0 and 
            i - last_swing_low_idx >= 3 and  # Minimum 3 bars for retracement
            close[i] > last_swing_low and    # Point 2: rally above Point 1
            np.min(low[max(0, i-3):i+1]) > last_swing_low):  # Point 3: fails to make new low
            bull_123[i] = True
        
        # Bearish 123: After swing high, price drops (Point 2), then fails to make new high (Point 3)
        if (last_swing_high_idx > 0 and 
            i - last_swing_high_idx >= 3 and  # Minimum 3 bars for retracement
            close[i] < last_swing_high and    # Point 2: drop below Point 1
            np.max(high[max(0, i-3):i+1]) < last_swing_high):  # Point 3: fails to make new high
            bear_123[i] = True
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(pp_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or
            np.isnan(s1_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        pp_val = pp_1w_aligned[i]
        r1_val = r1_1w_aligned[i]
        s1_val = s1_1w_aligned[i]
        
        # Weekly pivot bias: price above PP = bullish, below PP = bearish
        bullish_bias = price > pp_val
        bearish_bias = price < pp_val
        
        # Entry logic with weekly pivot filter
        if bull_123[i] and bullish_bias:
            signals[i] = 0.25
        elif bear_123[i] and bearish_bias:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_123Reversal_WeeklyPivotBias"
timeframe = "6h"
leverage = 1.0