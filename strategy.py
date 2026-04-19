#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Weekly Pivot Point Breakout with Volume Confirmation
# Long when: Price breaks above weekly R1 with volume > 2x 20-period average
# Short when: Price breaks below weekly S1 with volume > 2x 20-period average
# Exit when: Price returns to weekly pivot point
# Weekly Pivot = (Prior week High + Low + Close) / 3
# R1 = (2 * Pivot) - Prior week Low
# S1 = (2 * Pivot) - Prior week High
# Weekly pivot provides institutional support/resistance; breakouts with volume
# indicate institutional participation. Works in all regimes as it captures
# institutional breakouts rather than retail noise.
name = "6h_WeeklyPivot_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points using prior week's data
    # Pivot = (H + L + C) / 3
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = (2 * weekly_pivot) - weekly_low  # Resistance 1
    weekly_s1 = (2 * weekly_pivot) - weekly_high  # Support 1
    
    # Align weekly levels to 6h timeframe (wait for weekly bar to close)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        pivot = pivot_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Price breaks above weekly R1 with volume spike
            if price > r1 and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below weekly S1 with volume spike
            elif price < s1 and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to weekly pivot
            if price <= pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to weekly pivot
            if price >= pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals