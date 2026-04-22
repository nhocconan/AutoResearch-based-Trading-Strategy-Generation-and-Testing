#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction and volume confirmation
    # Weekly pivot from daily data provides institutional support/resistance levels
    # Breakouts in direction of weekly pivot bias have higher follow-through
    # Volume confirmation filters false breakouts
    # Works in bull/bear: buys strength in uptrend, sells weakness in downtrend
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from daily data (using prior week's data)
    # Weekly high = max of last 5 daily highs
    # Weekly low = min of last 5 daily lows  
    # Weekly close = last daily close
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().values
    
    # Pivot point = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Support 1 = (2 * P) - H
    weekly_s1 = (2 * weekly_pivot) - weekly_high
    # Resistance 1 = (2 * P) - L
    weekly_r1 = (2 * weekly_pivot) - weekly_low
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Donchian channel (20-period) on 6h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > 1.3 * vol_ma20  # Require 1.3x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high + price above weekly pivot + volume surge
            if close[i] > donchian_high[i] and close[i] > weekly_pivot_aligned[i] and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + price below weekly pivot + volume surge
            elif close[i] < donchian_low[i] and close[i] < weekly_pivot_aligned[i] and vol_surge[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses back below weekly pivot (for longs) or above weekly pivot (for shorts)
            if position == 1:
                if close[i] < weekly_pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > weekly_pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian_20_WeeklyPivot_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0