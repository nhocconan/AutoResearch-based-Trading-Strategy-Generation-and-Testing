#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout + volume spike + 1d choppiness regime filter.
Long when price breaks above Donchian(20) high with volume > 1.5x 20-period average and CHOP > 61.8 (range regime).
Short when price breaks below Donchian(20) low with volume > 1.5x 20-period average and CHOP > 61.8.
Exit when price touches opposite Donchian band or CHOP < 38.2 (trend regime).
Uses 1d for CHOP regime, 4h for Donchian and volume.
Target: 75-200 total trades over 4 years (19-50/year).
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
    
    # Get 1d data for choppiness regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR for choppiness
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    # Calculate 1d choppiness index
    def calculate_chop(high, low, close, atr_period=14, chop_period=14):
        atr = calculate_atr(high, low, close, atr_period)
        sum_atr = np.zeros_like(close)
        for i in range(len(close)):
            if i < chop_period:
                sum_atr[i] = np.nan
            else:
                sum_atr[i] = np.sum(atr[i-chop_period+1:i+1])
        highest_high = np.zeros_like(high)
        lowest_low = np.zeros_like(low)
        for i in range(len(high)):
            if i < chop_period:
                highest_high[i] = np.nan
                lowest_low[i] = np.nan
            else:
                highest_high[i] = np.max(high[i-chop_period+1:i+1])
                lowest_low[i] = np.min(low[i-chop_period+1:i+1])
        chop = np.zeros_like(close)
        for i in range(len(close)):
            if np.isnan(sum_atr[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or highest_high[i] == lowest_low[i]:
                chop[i] = np.nan
            else:
                chop[i] = 100 * np.log10(sum_atr[i] / (highest_high[i] - lowest_low[i])) / np.log10(chop_period)
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 4h Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = np.zeros_like(high)
        lower = np.zeros_like(low)
        for i in range(len(high)):
            if i < period - 1:
                upper[i] = np.nan
                lower[i] = np.nan
            else:
                upper[i] = np.max(high[i-period+1:i+1])
                lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # Calculate 4h volume spike (volume > 1.5x 20-period average)
    def calculate_volume_spike(volume, period=20, threshold=1.5):
        vol_ma = np.zeros_like(volume)
        for i in range(len(volume)):
            if i < period:
                vol_ma[i] = np.nan
            else:
                vol_ma[i] = np.mean(volume[i-period:i])
        spike = np.zeros_like(volume)
        for i in range(len(volume)):
            if np.isnan(vol_ma[i]):
                spike[i] = 0
            else:
                spike[i] = 1 if volume[i] > threshold * vol_ma[i] else 0
        return spike
    
    vol_spike = calculate_volume_spike(volume, 20, 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in range regime (CHOP > 61.8)
        in_range = chop_1d_aligned[i] > 61.8
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume spike and in range regime
            if close[i] > donch_upper[i] and vol_spike[i] and in_range:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower with volume spike and in range regime
            elif close[i] < donch_lower[i] and vol_spike[i] and in_range:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches Donchian lower OR regime shifts to trend (CHOP < 38.2)
            if close[i] < donch_lower[i] or chop_1d_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches Donchian upper OR regime shifts to trend (CHOP < 38.2)
            if close[i] > donch_upper[i] or chop_1d_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_VolumeSpike_CHOP_Range"
timeframe = "4h"
leverage = 1.0