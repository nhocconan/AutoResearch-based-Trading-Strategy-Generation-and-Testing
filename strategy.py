#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume confirmation
# Uses Donchian channels for structure, 1d ATR for volatility filter (proven from top performers),
# and volume spike for confirmation. Designed for 20-35 trades/year to minimize fee drag.
# Works in bull markets via breakout continuations and in bear markets via breakdown continuations.
# The 1d ATR provides a dynamic volatility filter that adapts to changing regimes while avoiding whipsaw.

name = "4h_Donchian20_1dATR_VolumeSpike_TrendFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) from prior completed 1d bar
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_1d = np.zeros_like(close_1d)
    atr14_1d[14:] = pd.Series(tr).rolling(window=14, min_periods=14).mean().values[13:]
    atr14_1d[:14] = np.nan
    atr14_1d_shifted = np.roll(atr14_1d, 1)
    atr14_1d_shifted[0] = np.nan
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d_shifted)
    
    # Calculate Donchian channels (20-period) from prior completed 4h bar
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    highest_20_shifted = np.roll(highest_20, 1)
    lowest_20_shifted = np.roll(lowest_20, 1)
    highest_20_shifted[0] = np.nan
    lowest_20_shifted[0] = np.nan
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(highest_20_shifted[i]) or
            np.isnan(lowest_20_shifted[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Dynamic ATR threshold: 1.5 * 1d ATR
        atr_threshold = 1.5 * atr14_1d_aligned[i]
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian AND above volatility filter AND volume spike
            if close[i] > highest_20_shifted[i] and close[i] > (close[i-1] + atr_threshold) and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian AND below volatility filter AND volume spike
            elif close[i] < lowest_20_shifted[i] and close[i] < (close[i-1] - atr_threshold) and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below lower Donchian OR below volatility filter
            if close[i] < lowest_20_shifted[i] or close[i] < (close[i-1] - atr_threshold):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above upper Donchian OR above volatility filter
            if close[i] > highest_20_shifted[i] or close[i] > (close[i-1] + atr_threshold):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals