#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter
# Uses 1-week pivot levels to filter breakout direction for trend alignment
# Weekly pivot provides higher timeframe bias to avoid counter-trend trades
# Target: 15-30 trades/year per symbol, works in bull/bear via weekly trend filter
# Breakout at Donchian(20) with weekly pivot alignment reduces false signals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-week data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot and levels from previous week
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    
    # Set first week values to NaN
    prev_close_1w[0] = np.nan
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    
    # Calculate weekly pivot points
    weekly_pivot = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    weekly_r1 = 2 * weekly_pivot - prev_low_1w
    weekly_s1 = 2 * weekly_pivot - prev_high_1w
    weekly_r2 = weekly_pivot + (prev_high_1w - prev_low_1w)
    weekly_s2 = weekly_pivot - (prev_high_1w - prev_low_1w)
    
    # Donchian channel (20-period) on 6h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high + price above weekly pivot + volume spike
            if (close[i] > high_20[i] and close[i] > weekly_pivot_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low + price below weekly pivot + volume spike
            elif (close[i] < low_20[i] and close[i] < weekly_pivot_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level
            if position == 1:
                if close[i] < low_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > high_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Volume_Filter"
timeframe = "6h"
leverage = 1.0