# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_WeeklyPivot_VWAP_Bounce
Hypothesis: Trade VWAP bounces on 6h timeframe with directional bias from weekly pivot. 
In bull markets (price above weekly pivot), look for long bounces off VWAP from below. 
In bear markets (price below weekly pivot), look for short bounces off VWAP from above.
Weekly pivot provides structural bias that works in both bull and bear markets. 
VWAP provides dynamic support/resistance that adapts to price action. 
Combined, they offer high-probability mean-reversion entries with trend alignment.
Target: 15-25 trades/year per symbol.
"""

name = "6h_WeeklyPivot_VWAP_Bounce"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot points (directional bias)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    high_prev_week = df_1w['high'].shift(1).values
    low_prev_week = df_1w['low'].shift(1).values
    close_prev_week = df_1w['close'].shift(1).values
    
    # Weekly pivot point calculation
    weekly_pivot = (high_prev_week + low_prev_week + close_prev_week) / 3.0
    weekly_r1 = 2 * weekly_pivot - low_prev_week
    weekly_s1 = 2 * weekly_pivot - high_prev_week
    
    # Align weekly levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Get daily data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate VWAP from daily data (typical price * volume)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_raw = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_values = vwap_raw.values
    
    # Align VWAP to 6h timeframe (no extra delay needed as VWAP is cumulative)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_values)
    
    # Get 6h data for price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.3x 20-period EMA (less strict than before)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly data (shifted) and VWAP calculation
    start_idx = 30  # reasonable warmup for VWAP and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or
            np.isnan(vwap_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly bias: price vs weekly pivot
        weekly_bias_up = close[i] > weekly_pivot_aligned[i]
        weekly_bias_down = close[i] < weekly_pivot_aligned[i]
        
        # VWAP bounce conditions
        vwap_bounce_long = (low[i] <= vwap_aligned[i] and close[i] > vwap_aligned[i])
        vwap_bounce_short = (high[i] >= vwap_aligned[i] and close[i] < vwap_aligned[i])
        
        if position == 0:
            # Long: bullish bias + VWAP bounce from below + volume
            if weekly_bias_up and vwap_bounce_long and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish bias + VWAP bounce from above + volume
            elif weekly_bias_down and vwap_bounce_short and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below VWAP or weekly S1
            if close[i] < vwap_aligned[i] or low[i] < weekly_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above VWAP or weekly R1
            if close[i] > vwap_aligned[i] or high[i] > weekly_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals