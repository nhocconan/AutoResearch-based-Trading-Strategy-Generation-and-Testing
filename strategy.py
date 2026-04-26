#!/usr/bin/env python3
"""
6h_WeeklyPivot_Reversal_With_Volume_Confirmation
Hypothesis: Weekly pivot points act as strong support/resistance levels. Price tends to reverse from weekly R2/S2 levels with volume confirmation. Works in both bull and bear markets as pivots adapt to price action. Uses 6h timeframe for lower frequency and reduced fee drag.
"""

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
    
    # Get weekly data for pivot points (using 1w HTF data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    # Using typical price: (H + L + C) / 3
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    prev_weekly_close = df_1w['close'].shift(1).values
    
    # Weekly pivot point
    weekly_pp = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    # Weekly R1, S1, R2, S2
    weekly_r1 = 2 * weekly_pp - prev_weekly_low
    weekly_s1 = 2 * weekly_pp - prev_weekly_high
    weekly_r2 = weekly_pp + (prev_weekly_high - prev_weekly_low)
    weekly_s2 = weekly_pp - (prev_weekly_high - prev_weekly_low)
    
    # Align weekly pivots to 6h timeframe
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of weekly pivot calculation (needs 1 bar), volume MA (20)
    start_idx = max(1, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_r2_aligned[i]) or 
            np.isnan(weekly_s2_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        weekly_r2_val = weekly_r2_aligned[i]
        weekly_s2_val = weekly_s2_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price touches or goes below S2 with volume spike (mean reversion from extreme)
            long_signal = (low_val <= weekly_s2_val) and (volume_val > 1.5 * vol_ma_val)
            # Short: price touches or goes above R2 with volume spike (mean reversion from extreme)
            short_signal = (high_val >= weekly_r2_val) and (volume_val > 1.5 * vol_ma_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price reaches weekly pivot point or weekly R1 (profit target)
            if (close_val >= weekly_pp_aligned[i]) or (close_val >= weekly_r1_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price reaches weekly pivot point or weekly S1 (profit target)
            if (close_val <= weekly_pp_aligned[i]) or (close_val <= weekly_s1_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Reversal_With_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0