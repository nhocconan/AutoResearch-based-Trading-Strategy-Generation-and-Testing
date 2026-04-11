#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly pivot levels and volume confirmation.
# Uses weekly pivot points (Pivot, R1, S1, R2, S2) calculated from prior week's range.
# Fades at S1/R1 (mean reversion) and breaks out at S2/R2 (momentum).
# Volume filter confirms institutional participation.
# Designed for 12-37 trades/year on 6h to minimize fee drift while capturing both mean reversion and breakout moves.
# Works in bull/bear markets by adapting to volatility regimes and using volume as confirmation.

name = "6h_1w_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's values for pivot calculation
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    
    # First week has no previous data
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate pivot point and levels
    # Pivot = (High + Low + Close) / 3
    # R1 = 2*Pivot - Low
    # S1 = 2*Pivot - High
    # R2 = Pivot + (High - Low)
    # S2 = Pivot - (High - Low)
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    # Calculate weekly average volume (10-period)
    volume_1w = df_1w['volume'].values
    vol_avg_10 = np.zeros_like(volume_1w, dtype=float)
    for i in range(9, len(volume_1w)):
        vol_avg_10[i] = np.mean(volume_1w[i-9:i+1])
    vol_avg_10[:9] = np.nan
    
    # Align weekly levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * weekly average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Fade at S1/R1 (mean reversion) - long at S1, short at R1
        fade_long = low[i] <= s1_aligned[i] and vol_filter
        fade_short = high[i] >= r1_aligned[i] and vol_filter
        
        # Breakout at S2/R2 (momentum) - long above R2, short below S2
        breakout_long = high[i] >= r2_aligned[i] and vol_filter
        breakout_short = low[i] <= s2_aligned[i] and vol_filter
        
        # Exit conditions: return to weekly pivot
        pivot_val = pivot[i] if i < len(pivot) else np.nan
        if not np.isnan(pivot_val):
            pivot_array = np.full_like(close_1w, pivot_val)
            pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_array)
            exit_long = not np.isnan(pivot_aligned[i]) and high[i] >= pivot_aligned[i]
            exit_short = not np.isnan(pivot_aligned[i]) and low[i] <= pivot_aligned[i]
        else:
            exit_long = exit_short = False
        
        # Priority: breakout > fade > hold
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif fade_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif fade_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and (exit_long or (low[i] >= s1_aligned[i] and high[i] <= r1_aligned[i])):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (exit_short or (low[i] >= s1_aligned[i] and high[i] <= r1_aligned[i])):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals