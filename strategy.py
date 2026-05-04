#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Uses Donchian channels from prior completed 6h bar for breakout structure
# Weekly pivot levels (from 1w data) determine trend: price above weekly pivot = long bias, below = short bias
# Volume confirmation (>2.0x 20 EMA) ensures breakout has participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Weekly pivot filter reduces whipsaw by aligning with higher timeframe structure
# Works in both bull and bear by following the weekly trend

name = "6h_Donchian20_WeeklyPivot_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard floor trader pivots)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    # Weekly R1: 2*P - L
    weekly_r1 = 2 * weekly_pivot - low_1w
    # Weekly S1: 2*P - H
    weekly_s1 = 2 * weekly_pivot - high_1w
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly bar close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Calculate Donchian channels (20-period) from prior completed 6h bar
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + price above weekly pivot + volume spike
            if close[i] > donchian_high[i] and close[i] > weekly_pivot_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + price below weekly pivot + volume spike
            elif close[i] < donchian_low[i] and close[i] < weekly_pivot_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR price crosses below weekly pivot
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2.0
            if not np.isnan(donchian_mid) and (close[i] < donchian_mid or close[i] < weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR price crosses above weekly pivot
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2.0
            if not np.isnan(donchian_mid) and (close[i] > donchian_mid or close[i] > weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals