#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d pivot direction (from prior day) and volume spike
    # Donchian channels provide clear breakout levels based on recent price extremes
    # 1-day pivot direction filters for institutional bias from previous session
    # Volume spike confirms institutional participation in the breakout
    # Works in bull/bear: breaks through key levels with trend and volume confirmation
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for pivot calculation and Donchian reference
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot points from previous day (standard floor trader pivots)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Pivot bias: bullish if close > pivot, bearish if close < pivot
    pivot_bias = np.where(close_1d > pivot, 1, np.where(close_1d < pivot, -1, 0))
    
    # Align pivot bias to 6h timeframe (using previous day's bias)
    pivot_bias_aligned = align_htf_to_ltf(prices, df_1d, pivot_bias)
    
    # Calculate Donchian channels (20-period) on 6h data
    # Using rolling window on high/low for breakout levels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(pivot_bias_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high with bullish pivot bias and volume spike
            if close[i] > donchian_high[i] and pivot_bias_aligned[i] > 0 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with bearish pivot bias and volume spike
            elif close[i] < donchian_low[i] and pivot_bias_aligned[i] < 0 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to opposite Donchian level or opposite pivot bias
            if position == 1:
                if close[i] < donchian_low[i] or pivot_bias_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_high[i] or pivot_bias_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian_20_Breakout_1dPivot_Direction_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0