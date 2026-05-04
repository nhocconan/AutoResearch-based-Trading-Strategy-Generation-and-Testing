#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d Williams %R extreme filter and volume confirmation
# Donchian breakouts capture momentum bursts; Williams %R > -20 or < -80 avoids overextended entries.
# Volume spike (>1.8x 20 EMA) confirms institutional participation. Works in bull/bear via breakout logic.
# Discrete sizing 0.25 limits risk. Target: 80-180 trades over 4 years (20-45/year).

name = "6h_Donchian20_1dWilliamsR_VolumeSpike"
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
    
    # Get 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Rolling window for highest high and lowest low (14-period)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    
    # Align 1d Williams %R to 6h timeframe (completed 1d bar only)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Donchian(20) on 6h timeframe
    highest_high_6h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_6h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(highest_high_6h[i]) or 
            np.isnan(lowest_low_6h[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.8 x 20-period EMA
        volume_confirm = volume[i] > (1.8 * vol_ema_20[i])
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper + not overbought + volume spike
            if close[i] > highest_high_6h[i] and williams_r_aligned[i] > -80 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower + not oversold + volume spike
            elif close[i] < lowest_low_6h[i] and williams_r_aligned[i] < -20 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR Williams %R becomes overbought OR volume drops
            donchian_mid = (highest_high_6h[i] + lowest_low_6h[i]) / 2.0
            if (close[i] < donchian_mid or 
                williams_r_aligned[i] < -90 or  # exited overbought territory
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR Williams %R becomes oversold OR volume drops
            donchian_mid = (highest_high_6h[i] + lowest_low_6h[i]) / 2.0
            if (close[i] > donchian_mid or 
                williams_r_aligned[i] > -10 or  # exited oversold territory
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals