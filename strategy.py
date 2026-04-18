#!/usr/bin/env python3
"""
6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation
Long when price breaks above Donchian(20) high and weekly pivot indicates uptrend.
Short when price breaks below Donchian(20) low and weekly pivot indicates downtrend.
Uses volume confirmation (2x average volume) to filter false breakouts.
Designed for low trade frequency with clear trend-following edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot direction (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Align weekly pivots to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Weekly trend: price > R3 = uptrend, price < S3 = downtrend
    # Note: We use the aligned values from previous bar to avoid look-ahead
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 40  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(pivot_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend direction (using previous bar's aligned values to avoid look-ahead)
        prev_pivot = pivot_1w_aligned[i-1] if i > 0 else pivot_1w_aligned[i]
        prev_r3 = r3_1w_aligned[i-1] if i > 0 else r3_1w_aligned[i]
        prev_s3 = s3_1w_aligned[i-1] if i > 0 else s3_1w_aligned[i]
        
        weekly_uptrend = close[i-1] > prev_r3 if i > 0 else close[i] > prev_r3
        weekly_downtrend = close[i-1] < prev_s3 if i > 0 else close[i] < prev_s3
        
        price = close[i]
        
        if position == 0:
            # Long: breakout above Donchian high + weekly uptrend + volume spike
            if (price > donchian_high[i] and weekly_uptrend and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low + weekly downtrend + volume spike
            elif (price < donchian_low[i] and weekly_downtrend and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: breakdown below Donchian low or weekly trend turns down
            if price < donchian_low[i] or (i > 0 and close[i-1] < s3_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: breakout above Donchian high or weekly trend turns up
            if price > donchian_high[i] or (i > 0 and close[i-1] > r3_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0