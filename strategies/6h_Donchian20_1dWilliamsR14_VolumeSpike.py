#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d Williams %R(14) regime filter and volume spike confirmation
# Donchian breakout provides clear structure-based entries in trending markets
# 1d Williams %R(14) avoids extremes: long only when %R > -80 (not oversold), short only when %R < -20 (not overbought)
# Volume spike (>2.0x 20-period EMA volume) confirms institutional participation
# Discrete sizing 0.28 targets 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Works in bull markets (breakouts with favorable regime) and bear markets (breakouts with favorable regime)
# Williams %R acts as a regime filter to avoid counter-trend trades at extremes

name = "6h_Donchian20_1dWilliamsR14_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R(14) regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need enough data for Williams %R calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R(14) from prior completed 1d bar
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_14 = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r_14_shifted = np.roll(williams_r_14, 1)
    williams_r_14_shifted[0] = np.nan
    
    # Align HTF indicator to 6h timeframe (wait for completed 1d bar)
    williams_r_14_aligned = align_htf_to_ltf(prices, df_1d, williams_r_14_shifted)
    
    # Calculate Donchian channels (20-period) from prior completed 6h bar
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    highest_high_20_shifted = np.roll(highest_high_20, 1)
    lowest_low_20_shifted = np.roll(lowest_low_20, 1)
    highest_high_20_shifted[0] = np.nan
    lowest_low_20_shifted[0] = np.nan
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_14_aligned[i]) or np.isnan(highest_high_20_shifted[i]) or 
            np.isnan(lowest_low_20_shifted[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band AND Williams %R > -80 (not oversold) AND volume spike
            if close[i] > highest_high_20_shifted[i] and williams_r_14_aligned[i] > -80 and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.28
                position = 1
            # Short conditions: price breaks below Donchian lower band AND Williams %R < -20 (not overbought) AND volume spike
            elif close[i] < lowest_low_20_shifted[i] and williams_r_14_aligned[i] < -20 and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.28
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian middle OR Williams %R crosses above -20 (overbought)
            donchian_middle = (highest_high_20_shifted[i] + lowest_low_20_shifted[i]) / 2.0
            if close[i] < donchian_middle or williams_r_14_aligned[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        elif position == -1:
            # Exit short: price closes above Donchian middle OR Williams %R crosses below -80 (oversold)
            donchian_middle = (highest_high_20_shifted[i] + lowest_low_20_shifted[i]) / 2.0
            if close[i] > donchian_middle or williams_r_14_aligned[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals