#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with weekly pivot direction and volume confirmation.
# Uses 1-week timeframe to determine trend direction via weekly pivot levels (R1/S1).
# Breakouts in direction of weekly trend are taken with volume confirmation.
# Designed for 12h timeframe to capture multi-day swings with low frequency.
# Target: 12-37 trades/year per symbol (48-148 total) to minimize fee drift.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-week data for trend filter via pivot levels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using prior week's data)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # We use the prior week's data, so we shift by 1
    pivot_1w = (np.roll(high_1w, 1) + np.roll(low_1w, 1) + np.roll(close_1w, 1)) / 3
    r1_1w = 2 * pivot_1w - np.roll(low_1w, 1)
    s1_1w = 2 * pivot_1w - np.roll(high_1w, 1)
    
    # Trend filter: price above R1 = bullish, below S1 = bearish
    trend_bullish = close_1w > r1_1w
    trend_bearish = close_1w < s1_1w
    
    # Load 1-day data for Donchian channel (using prior day's data to avoid look-ahead)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian channel: 20-period high/low (prior period)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Shift to use only completed periods (avoid look-ahead)
    high_20 = np.roll(high_20, 1)
    low_20 = np.roll(low_20, 1)
    
    # Volume spike filter (12-period on 12h)
    vol_ma12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    vol_spike = volume > 2.0 * vol_ma12
    
    # Align indicators to 12-hour timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, trend_bullish.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, trend_bearish.astype(float))
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or
            np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(vol_ma12[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + weekly bullish trend + volume spike
            if (close[i] > high_20_aligned[i] and 
                trend_bullish_aligned[i] > 0.5 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + weekly bearish trend + volume spike
            elif (close[i] < low_20_aligned[i] and 
                  trend_bearish_aligned[i] > 0.5 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price breaks opposite Donchian level or trend changes
            if position == 1:
                if (close[i] < low_20_aligned[i] or trend_bullish_aligned[i] <= 0.5):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > high_20_aligned[i] or trend_bearish_aligned[i] <= 0.5):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_WeeklyPivot_Trend_Volume_Spike"
timeframe = "12h"
leverage = 1.0