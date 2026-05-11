#!/usr/bin/env python3
name = "6h_WeeklyPivot_Trend_Filter_v3"
timeframe = "6h"
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
    
    # Calculate weekly pivot (using weekly high/low/close from 1w data)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point and key levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Trend filter: 6-period EMA on 6h close (fast trend)
    ema_fast = pd.Series(close).ewm(span=6, min_periods=6, adjust=False).mean().values
    # Slow trend: 24-period EMA on 6h close (4-day trend)
    ema_slow = pd.Series(close).ewm(span=24, min_periods=24, adjust=False).mean().values
    
    # Volume filter: 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or 
            np.isnan(s2_1w_aligned[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_pivot = close[i] > pivot_1w_aligned[i]
        price_below_pivot = close[i] < pivot_1w_aligned[i]
        price_above_r1 = close[i] > r1_1w_aligned[i]
        price_below_s1 = close[i] < s1_1w_aligned[i]
        price_above_r2 = close[i] > r2_1w_aligned[i]
        price_below_s2 = close[i] < s2_1w_aligned[i]
        ema_fast_above_slow = ema_fast[i] > ema_slow[i]
        ema_fast_below_slow = ema_fast[i] < ema_slow[i]
        
        if position == 0:
            # Long: Price above weekly R1 + fast EMA above slow EMA + volume spike
            if price_above_r1 and ema_fast_above_slow and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly S1 + fast EMA below slow EMA + volume spike
            elif price_below_s1 and ema_fast_below_slow and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: Price crosses below weekly pivot OR trend reverses
                if price_below_pivot or ema_fast_below_slow:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: Price crosses above weekly pivot OR trend reverses
                if price_above_pivot or ema_fast_above_slow:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals