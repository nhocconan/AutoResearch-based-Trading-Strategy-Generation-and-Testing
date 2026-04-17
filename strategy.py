#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d volume spike + 1d choppiness regime filter.
Long when price breaks above Donchian(20) high with volume spike and chop > 61.8 (range regime).
Short when price breaks below Donchian(20) low with volume spike and chop > 61.8.
Exit when price breaks opposite Donchian level or chop < 38.2 (trend regime).
Uses 12h for price/Donchian, 1d for volume and chop filter.
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
    
    # Get 1d data for volume and chop filters
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(14) for chop calculation
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
    def calculate_chop(high, low, close, period=14):
        atr = calculate_atr(high, low, close, period)
        sum_atr = np.zeros_like(close)
        sum_atr[period] = np.sum(atr[1:period+1])
        for i in range(period+1, len(close)):
            sum_atr[i] = sum_atr[i-1] - atr[i-period] + atr[i]
        
        highest_high = np.zeros_like(high)
        lowest_low = np.zeros_like(low)
        highest_high[period] = np.max(high[1:period+1])
        lowest_low[period] = np.min(low[1:period+1])
        for i in range(period+1, len(close)):
            highest_high[i] = max(highest_high[i-1], high[i])
            lowest_low[i] = min(lowest_low[i-1], low[i])
        
        range_hl = highest_high - lowest_low
        chop = np.zeros_like(close)
        for i in range(period, len(close)):
            if range_hl[i] > 0:
                chop[i] = 100 * np.log10(sum_atr[i] / range_hl[i]) / np.log10(period)
            else:
                chop[i] = 50  # neutral when no range
        return chop
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period average)
    def calculate_volume_spike(volume, period=20):
        vol_ma = np.zeros_like(volume)
        for i in range(period, len(volume)):
            vol_ma[i] = np.mean(volume[i-period:i])
        vol_spike = np.zeros_like(volume)
        vol_spike[period:] = volume[period:] > (2.0 * vol_ma[period:])
        return vol_spike
    
    # Calculate 1d indicators
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    vol_spike_1d = calculate_volume_spike(volume_1d, 20)
    
    # Align 1d indicators
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Calculate 12h Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = np.zeros_like(high)
        lower = np.zeros_like(low)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(chop_1d_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = vol_spike_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        
        # Regime filter: only trade in range regime (chop > 61.8)
        is_range = chop_val > 61.8
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume spike in range regime
            if price > upper and vol_spike and is_range:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower with volume spike in range regime
            elif price < lower and vol_spike and is_range:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian lower OR chop < 38.2 (trend regime)
            if price < lower or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian upper OR chop < 38.2 (trend regime)
            if price > upper or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_ChopRegime"
timeframe = "12h"
leverage = 1.0