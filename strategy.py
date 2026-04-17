#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d volume spike + chop regime filter.
Long when price breaks above Donchian upper band with volume > 1.5x 20-period average and chop > 61.8 (range regime).
Short when price breaks below Donchian lower band with volume > 1.5x 20-period average and chop > 61.8.
Exit when price touches opposite Donchian band or chop < 38.2 (trend regime).
Uses 1d for volume and chop filters, 12h for Donchian channels.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for volume and chop filters
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    def donchian_channels(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_12h, lower_12h = donchian_channels(high_12h, low_12h, 20)
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    
    # Calculate 1d volume spike filter (volume > 1.5x 20-period average)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (vol_ma_1d * 1.5)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Calculate 1d Choppiness Index (CHOP) - range regime filter
    def choppiness_index(high, low, close, period=14):
        atr = np.zeros_like(high)
        for i in range(1, len(high)):
            atr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing for ATR
        atr_smoothed = np.zeros_like(atr)
        atr_smoothed[period] = np.mean(atr[1:period+1])
        for i in range(period+1, len(atr)):
            atr_smoothed[i] = (atr_smoothed[i-1] * (period-1) + atr[i]) / period
        
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        
        chop = np.zeros_like(high)
        for i in range(period, len(high)):
            if atr_smoothed[i] > 0 and (highest_high[i] - lowest_low[i]) > 0:
                chop[i] = 100 * np.log10(atr_smoothed[i] * period / (highest_high[i] - lowest_low[i])) / np.log10(period)
            else:
                chop[i] = 50  # neutral
        return chop
    
    chop_1d = choppiness_index(high_1d, low_1d, close_1d, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_12h_aligned[i]) or 
            np.isnan(lower_12h_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = vol_spike_1d_aligned[i] > 0.5
        chop_val = chop_1d_aligned[i]
        
        # Regime: range when CHOP > 61.8
        is_range = chop_val > 61.8
        
        if position == 0:
            # Long: price > upper Donchian + volume spike + range regime
            if price > upper_12h_aligned[i] and vol_spike and is_range:
                signals[i] = 0.25
                position = 1
            # Short: price < lower Donchian + volume spike + range regime
            elif price < lower_12h_aligned[i] and vol_spike and is_range:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < lower Donchian OR regime shifts to trend (CHOP < 38.2)
            if price < lower_12h_aligned[i] or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > upper Donchian OR regime shifts to trend (CHOP < 38.2)
            if price > upper_12h_aligned[i] or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_ChopRegime"
timeframe = "12h"
leverage = 1.0