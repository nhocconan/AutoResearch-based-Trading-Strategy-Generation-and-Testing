#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d volume spike + choppiness regime filter.
Long when price breaks above Donchian(20) high with volume > 1.5x 20-period average and CHOP > 61.8 (range regime).
Short when price breaks below Donchian(20) low with volume > 1.5x 20-period average and CHOP > 61.8.
Exit when price reverts to Donchian(20) midpoint or volume drops below average.
Uses 1d for volume spike and CHOP regime, 12h for Donchian channels.
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
    
    # Get 1d data for volume and choppiness regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR for choppiness
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        tr[0] = high[0] - low[0]
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
        
        max_high = np.zeros_like(high)
        min_low = np.zeros_like(low)
        for i in range(len(high)):
            if i < chop_period:
                max_high[i] = np.nan
                min_low[i] = np.nan
            else:
                max_high[i] = np.max(high[i-chop_period+1:i+1])
                min_low[i] = np.min(low[i-chop_period+1:i+1])
        
        chop = np.zeros_like(close)
        for i in range(len(close)):
            if np.isnan(sum_atr[i]) or np.isnan(max_high[i]) or np.isnan(min_low[i]) or max_high[i] == min_low[i]:
                chop[i] = np.nan
            else:
                chop[i] = 100 * np.log10(sum_atr[i] / (max_high[i] - min_low[i])) / np.log10(chop_period)
        return chop
    
    # Calculate 1d volume spike (volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)
    
    # Calculate 1d choppiness
    chop = calculate_chop(high_1d, low_1d, close_1d, 14, 14)
    chop_regime = chop > 61.8  # range regime
    
    # Align 1d indicators
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime.astype(float))
    
    # Calculate 12h Donchian channels
    def donchian_channels(high, low, period=20):
        upper = np.zeros_like(high)
        lower = np.zeros_like(high)
        mid = np.zeros_like(high)
        for i in range(len(high)):
            if i < period:
                upper[i] = np.nan
                lower[i] = np.nan
                mid[i] = np.nan
            else:
                upper[i] = np.max(high[i-period+1:i+1])
                lower[i] = np.min(low[i-period+1:i+1])
                mid[i] = (upper[i] + lower[i]) / 2
        return upper, lower, mid
    
    upper_20, lower_20, mid_20 = donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime and volume conditions
        vol_spike = volume_spike_aligned[i] > 0.5  # boolean as float
        chop_range = chop_regime_aligned[i] > 0.5   # boolean as float
        
        if position == 0:
            # Long: price > upper Donchian + volume spike + chop regime (range)
            if close[i] > upper_20[i] and vol_spike and chop_range:
                signals[i] = 0.25
                position = 1
            # Short: price < lower Donchian + volume spike + chop regime (range)
            elif close[i] < lower_20[i] and vol_spike and chop_range:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < midpoint OR volume drops below average OR chop regime ends
            if close[i] < mid_20[i] or not vol_spike or not chop_range:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > midpoint OR volume drops below average OR chop regime ends
            if close[i] > mid_20[i] or not vol_spike or not chop_range:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_ChopRegime"
timeframe = "12h"
leverage = 1.0