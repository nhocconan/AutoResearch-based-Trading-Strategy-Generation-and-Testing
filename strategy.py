# 1d_WeeklyPivot_S1S2_R1R2_Breakout_Volume
# Hypothesis: Weekly pivot levels provide strong institutional support/resistance.
# Price breaking above R1/R2 with volume confirms bullish momentum; breaking below S1/S2 confirms bearish momentum.
# Weekly timeframe reduces noise and aligns with institutional order flow.
# Works in both bull and bear markets by capturing breakouts from key weekly levels.
# Uses volume confirmation to avoid false breakouts.
# Target: 20-50 trades/year on 1d timeframe to minimize fee drag.

#!/usr/bin/env python3
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
    
    # Get weekly data for pivot levels
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot points (standard formula)
    pivot_weekly = (high_weekly + low_weekly + close_weekly) / 3.0
    range_weekly = high_weekly - low_weekly
    r1_weekly = 2 * pivot_weekly - low_weekly
    s1_weekly = 2 * pivot_weekly - high_weekly
    r2_weekly = pivot_weekly + range_weekly
    s2_weekly = pivot_weekly - range_weekly
    
    # Align weekly pivot levels to daily
    pivot_weekly_aligned = align_htf_to_ltf(prices, df_weekly, pivot_weekly)
    r1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s1_weekly)
    r2_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r2_weekly)
    s2_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s2_weekly)
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_weekly_aligned[i]) or np.isnan(r1_weekly_aligned[i]) or np.isnan(s1_weekly_aligned[i]) or
            np.isnan(r2_weekly_aligned[i]) or np.isnan(s2_weekly_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 or R2 with volume
            if (close[i] > r1_weekly_aligned[i] or close[i] > r2_weekly_aligned[i]) and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 or S2 with volume
            elif (close[i] < s1_weekly_aligned[i] or close[i] < s2_weekly_aligned[i]) and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below pivot
            if close[i] < pivot_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above pivot
            if close[i] > pivot_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_S1S2_R1R2_Breakout_Volume"
timeframe = "1d"
leverage = 1.0