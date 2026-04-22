#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction and volume confirmation
    # Donchian breakouts capture momentum in trending markets
    # Weekly pivot direction from 1d timeframe filters for institutional bias
    # Volume spike (2x 20-period MA) confirms institutional participation
    # Works in both bull and bear: breakouts in trends, pivot direction avoids counter-trend entries
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from prior week's OHLC
    # Need at least 5 days for weekly calculation
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Get prior week's high, low, close for pivot calculation
    # We'll use rolling window of 5 days (1 week) to get weekly OHLC
    week_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1)  # Prior week
    week_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1)
    week_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(1)
    
    # Calculate pivot points
    pivot = (week_high + week_low + week_close) / 3
    r1 = 2 * pivot - week_low
    s1 = 2 * pivot - week_high
    r2 = pivot + (week_high - week_low)
    s2 = pivot - (week_high - week_low)
    r3 = week_high + 2 * (pivot - week_low)
    s3 = week_low - 2 * (week_high - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2.values)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    
    # Donchian channel (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high + price above weekly pivot + volume spike
            if close[i] > donchian_high[i] and close[i] > pivot_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low + price below weekly pivot + volume spike
            elif close[i] < donchian_low[i] and close[i] < pivot_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to Donchian midpoint or opposite pivot level
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if position == 1:
                if close[i] < donchian_mid:  # Return to mean
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_mid:  # Return to mean
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian_20_Breakout_WeeklyPivot_Direction_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0