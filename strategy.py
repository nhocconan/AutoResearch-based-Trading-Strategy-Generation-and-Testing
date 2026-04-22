#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with daily pivot level confirmation and volume spike.
# Uses daily pivot levels (R1/S1) to define key support/resistance zones.
# Breakouts above daily R1 or below daily S1 are taken only with volume confirmation.
# Designed for 12h timeframe to capture multi-day swings with low frequency (<30 trades/year).
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for pivot levels and Donchian channel
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (using prior day's data)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    pivot_1d = (np.roll(high_1d, 1) + np.roll(low_1d, 1) + np.roll(close_1d, 1)) / 3
    r1_1d = 2 * pivot_1d - np.roll(low_1d, 1)
    s1_1d = 2 * pivot_1d - np.roll(high_1d, 1)
    
    # Donchian channel: 20-period high/low (prior period)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Shift to use only completed periods (avoid look-ahead)
    high_20 = np.roll(high_20, 1)
    low_20 = np.roll(low_20, 1)
    
    # Volume spike filter (12-period on 12h: 6 days)
    vol_ma12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    vol_spike = volume > 2.0 * vol_ma12
    
    # Align indicators to 12-hour timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(vol_ma12[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above daily R1 + Donchian high + volume spike
            if (close[i] > r1_1d_aligned[i] and 
                close[i] > high_20_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily S1 + Donchian low + volume spike
            elif (close[i] < s1_1d_aligned[i] and 
                  close[i] < low_20_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price breaks opposite pivot level or Donchian level
            if position == 1:
                if (close[i] < s1_1d_aligned[i] or close[i] < low_20_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > r1_1d_aligned[i] or close[i] > high_20_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_DailyPivot_Level_Volume_Spike"
timeframe = "12h"
leverage = 1.0