#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
# Uses Donchian channel breakouts for trend capture, weekly pivot points to determine
# the dominant trend direction, and volume to confirm breakout strength. Works in
# both bull and bear by only taking breakouts aligned with the weekly pivot trend.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data (primary timeframe) for price action
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Load weekly data for pivot points and trend direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period) on 6h
    donch_high_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donch_low_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly pivot points (standard floor trader pivots)
    # Pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    # Support 1 = (2 * Pivot) - High
    s1_1w = (2 * pivot_1w) - high_1w
    # Resistance 1 = (2 * Pivot) - Low
    r1_1w = (2 * pivot_1w) - low_1w
    
    # Weekly trend: bullish if close > pivot, bearish if close < pivot
    weekly_trend_bull = close_1w > pivot_1w
    weekly_trend_bear = close_1w < pivot_1w
    
    # Volume average (20-period on 6h)
    vol_avg_6h = pd.Series(df_6h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    donch_high_6h_aligned = align_htf_to_ltf(prices, df_6h, donch_high_6h)
    donch_low_6h_aligned = align_htf_to_ltf(prices, df_6h, donch_low_6h)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    weekly_trend_bull_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bull.astype(float))
    weekly_trend_bear_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bear.astype(float))
    vol_avg_aligned = align_htf_to_ltf(prices, df_6h, vol_avg_6h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_6h_aligned[i]) or np.isnan(donch_low_6h_aligned[i]) or
            np.isnan(pivot_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(r1_1w_aligned[i]) or np.isnan(weekly_trend_bull_aligned[i]) or
            np.isnan(weekly_trend_bear_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + weekly trend bullish + volume spike
        if (close[i] > donch_high_6h_aligned[i] and
            weekly_trend_bull_aligned[i] > 0.5 and  # Weekly trend is bullish
            volume[i] > 1.5 * vol_avg_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + weekly trend bearish + volume spike
        elif (close[i] < donch_low_6h_aligned[i] and
              weekly_trend_bear_aligned[i] > 0.5 and  # Weekly trend is bearish
              volume[i] > 1.5 * vol_avg_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or price crosses weekly pivot (trend change)
        elif position == 1 and (close[i] < donch_low_6h_aligned[i] or 
                                close[i] < pivot_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donch_high_6h_aligned[i] or 
                                 close[i] > pivot_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0