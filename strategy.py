# 6h Weekly Pivot Reversion with Volume Confirmation
# Hypothesis: Weekly pivots act as strong support/resistance. Price tends to revert from weekly R3/S3 levels
# back toward the weekly pivot point, especially on volume expansion. Works in both trending and ranging markets
# as reversals at extremes. Weekly timeframe provides structural context for 6s entries.
# Target: 12-37 trades/year (50-150 over 4 years) with size 0.25

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_R3S3_Reversion_Volume"
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
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    prev_weekly_high = df_weekly['high'].shift(1).values
    prev_weekly_low = df_weekly['low'].shift(1).values
    prev_weekly_close = df_weekly['close'].shift(1).values
    
    # Weekly pivot point
    pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    
    # Weekly R3 and S3 levels
    r3 = pivot + 2.0 * (prev_weekly_high - prev_weekly_low)
    s3 = pivot - 2.0 * (prev_weekly_high - prev_weekly_low)
    
    # Volume filter: current 6h volume > 1.8 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    # Align weekly levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_weekly, pivot)
    r3_6h = align_htf_to_ltf(prices, df_weekly, r3)
    s3_6h = align_htf_to_ltf(prices, df_weekly, s3)
    volume_filter_6h = align_htf_to_ltf(prices, df_weekly, volume_filter)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_6h[i]) or np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(volume_filter_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot_val = pivot_6h[i]
        r3_val = r3_6h[i]
        s3_val = s3_6h[i]
        vol_filter = volume_filter_6h[i]
        
        if position == 0:
            # Enter long: price at or below S3 with volume expansion (mean reversion long)
            if close[i] <= s3_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: price at or above R3 with volume expansion (mean reversion short)
            elif close[i] >= r3_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses above weekly pivot (mean reversion complete)
            if close[i] > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below weekly pivot (mean reversion complete)
            if close[i] < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals