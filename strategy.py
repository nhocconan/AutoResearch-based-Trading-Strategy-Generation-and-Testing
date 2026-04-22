#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with daily pivot direction and volume confirmation.
# Uses daily timeframe to determine trend direction via daily pivot levels (R1/S1).
# Breakouts in direction of daily trend are taken with volume confirmation.
# Designed for 12h timeframe to capture multi-day swings with low frequency.
# Target: 15-30 trades/year per symbol (60-120 total) to minimize fee drift.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for trend filter via pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (using prior day's data)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # We use the prior day's data, so we shift by 1
    pivot_1d = (np.roll(high_1d, 1) + np.roll(low_1d, 1) + np.roll(close_1d, 1)) / 3
    r1_1d = 2 * pivot_1d - np.roll(low_1d, 1)
    s1_1d = 2 * pivot_1d - np.roll(high_1d, 1)
    
    # Trend filter: price above R1 = bullish, below S1 = bearish
    trend_bullish = close_1d > r1_1d
    trend_bearish = close_1d < s1_1d
    
    # Load 12h data for Donchian channel (using prior period's data to avoid look-ahead)
    # For 12h, we need to use the same timeframe as prices for the Donchian calculation
    # Since we're using 12h timeframe, we calculate Donchian on the 12h data directly
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Shift to use only completed periods (avoid look-ahead)
    high_20 = np.roll(high_20, 1)
    low_20 = np.roll(low_20, 1)
    
    # Volume spike filter (10-period on 12h)
    vol_ma10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_spike = volume > 2.0 * vol_ma10
    
    # Align indicators to 12h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(vol_ma10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + daily bullish trend + volume spike
            if (close[i] > high_20[i] and 
                trend_bullish_aligned[i] > 0.5 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + daily bearish trend + volume spike
            elif (close[i] < low_20[i] and 
                  trend_bearish_aligned[i] > 0.5 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price breaks opposite Donchian level or trend changes
            if position == 1:
                if (close[i] < low_20[i] or trend_bullish_aligned[i] <= 0.5):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > high_20[i] or trend_bearish_aligned[i] <= 0.5):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_DailyPivot_Trend_Volume_Spike"
timeframe = "12h"
leverage = 1.0