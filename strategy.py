#!/usr/bin/env python3
"""
6h_24h_Donchian_20_WeeklyPivotDirection_VolumeConfirm
Hypothesis: Uses 6h Donchian breakout (20-period) confirmed by weekly pivot direction (1d data) and volume spikes. 
Weekly pivot provides directional bias from higher timeframe, reducing false breakouts in ranging markets. 
Designed to work in both bull and bear markets by following the weekly trend direction while using volume 
confirmation to avoid low-probability breakouts. Targets low trade frequency (12-30/year) for minimal fee drag.
"""

name = "6h_24h_Donchian_20_WeeklyPivotDirection_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 6h Donchian Channels (20-period) ---
    # Use rolling window with min_periods to avoid look-ahead
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # --- 1d OHLCV for Weekly Pivot Calculation ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot from prior week's daily data
    # We need 5 trading days for a week
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get last 5 days (prior complete week)
    # We'll use the most recent 5 days available (excluding current forming day)
    # Since we want prior week's data, we look back 5 days from yesterday
    # But for simplicity and to avoid look-ahead, we use the last completed week
    # We'll calculate pivot using the last 5 days where we have complete data
    # For each 6h bar, we need the weekly pivot from the most recent completed week
    
    # Pre-calculate weekly pivot values for each day
    # We need at least 5 days of data
    weekly_pivot = np.full(len(df_1d), np.nan)
    weekly_R1 = np.full(len(df_1d), np.nan)
    weekly_S1 = np.full(len(df_1d), np.nan)
    
    for i in range(4, len(df_1d)):  # Start from index 4 (5th day, 0-indexed)
        # Use previous 5 days (i-4 to i) to calculate pivot for day i
        # But we want the pivot for the week ending at day i-1 (prior complete week)
        # So we use days i-5 to i-1
        if i-5 < 0:
            continue
            
        week_high = np.max(high_1d[i-5:i])  # Days i-5 to i-1
        week_low = np.min(low_1d[i-5:i])
        week_close = close_1d[i-1]  # Close of day i-1 (last day of prior week)
        
        pivot_val = (week_high + week_low + week_close) / 3.0
        range_val = week_high - week_low
        
        # Weekly pivot levels (using standard pivot, not camarilla)
        weekly_pivot[i-1] = pivot_val  # Assign to the last day of the week
        weekly_R1[i-1] = pivot_val + (range_val * 1.1 / 12)
        weekly_S1[i-1] = pivot_val - (range_val * 1.1 / 12)
    
    # Align weekly pivot data to 6h timeframe
    # We want the weekly pivot value from the most recent completed week
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_R1_aligned = align_htf_to_ltf(prices, df_1d, weekly_R1)
    weekly_S1_aligned = align_htf_to_ltf(prices, df_1d, weekly_S1)
    
    # --- Volume Spike Detection (24-period average on 6h) ---
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Donchian and volume)
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_R1_aligned[i]) or 
            np.isnan(weekly_S1_aligned[i]) or np.isnan(vol_ratio[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 2.0
        
        # Determine weekly bias: 
        # If price above weekly pivot, bias is long; below pivot, bias is short
        weekly_bias_long = close[i] > weekly_pivot_aligned[i]
        weekly_bias_short = close[i] < weekly_pivot_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and weekly bias long
            if (close[i] > donchian_high[i] and 
                volume_spike and 
                weekly_bias_long):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and weekly bias short
            elif (close[i] < donchian_low[i] and 
                  volume_spike and 
                  weekly_bias_short):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Donchian breakout or loss of weekly bias
            if position == 1:
                # Exit long: price breaks below Donchian low OR weekly bias turns short
                if (close[i] < donchian_low[i] or 
                    weekly_bias_short):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above Donchian high OR weekly bias turns long
                if (close[i] > donchian_high[i] or 
                    weekly_bias_long):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals