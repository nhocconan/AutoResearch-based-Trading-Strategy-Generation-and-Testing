#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian breakout with 1d weekly pivot context and volume confirmation
    # Uses weekly pivot points from daily data to establish institutional support/resistance
    # Breakouts from weekly pivot levels with volume confirmation capture institutional flow
    # Works in both bull and bear: longs from weekly support, shorts from weekly resistance
    
    # Load daily data once for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from daily data (using prior week's data)
    # We'll use a rolling window of 5 days (1 week) to calculate weekly pivot
    def calculate_weekly_pivot(high, low, close):
        # Weekly high, low, close from prior 5 days
        weekly_high = pd.Series(high).rolling(window=5, min_periods=5).max().shift(1)  # prior week
        weekly_low = pd.Series(low).rolling(window=5, min_periods=5).min().shift(1)
        weekly_close = pd.Series(close).rolling(window=5, min_periods=5).last().shift(1)
        
        # Pivot point calculation
        pivot = (weekly_high + weekly_low + weekly_close) / 3
        r1 = 2 * pivot - weekly_low
        s1 = 2 * pivot - weekly_high
        r2 = pivot + (weekly_high - weekly_low)
        s2 = pivot - (weekly_high - weekly_low)
        r3 = weekly_high + 2 * (pivot - weekly_low)
        s3 = weekly_low - 2 * (weekly_high - pivot)
        
        return pivot.values, r1.values, r2.values, r3.values, s1.values, s2.values, s3.values
    
    pivot, r1, r2, r3, s1, s2, s3 = calculate_weekly_pivot(high_1d, low_1d, close_1d)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    
    # 6h Donchian channel (20 periods)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period surge)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.8 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(pivot_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout above weekly R3 with volume surge
            if close[i] > donchian_high[i] and close[i] > r3_6h[i] and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below weekly S3 with volume surge
            elif close[i] < donchian_low[i] and close[i] < s3_6h[i] and vol_surge[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to weekly pivot or opposite Donchian band
            if position == 1:
                if close[i] < pivot_6h[i] or close[i] < donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > pivot_6h[i] or close[i] > donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian_Breakout_WeeklyPivot_R3S3_VolumeSurge_v1"
timeframe = "6h"
leverage = 1.0