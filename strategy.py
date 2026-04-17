#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d volume spike + 1d chop regime filter.
Long when price breaks above Donchian(20) high with volume > 1.5x 20-period average and CHOP > 61.8 (range regime).
Short when price breaks below Donchian(20) low with volume > 1.5x 20-period average and CHOP > 61.8.
Exit when price crosses Donchian midpoint or CHOP < 38.2 (trend regime).
Uses 1d for Donchian, volume, and CHOP to avoid lower timeframe noise.
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
    
    # Get 1d data for Donchian, volume, and CHOP
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        middle = (upper + lower) / 2
        return upper, lower, middle
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (14-period)
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(high)
        for i in range(1, len(high)):
            atr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing for ATR
        atr_sum = np.zeros_like(atr)
        atr_sum[period] = np.sum(atr[1:period+1])
        for i in range(period+1, len(atr)):
            atr_sum[i] = atr_sum[i-1] - (atr_sum[i-1] / period) + atr[i]
        
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        
        chop = np.zeros_like(close)
        for i in range(period, len(close)):
            if atr_sum[i] > 0 and highest_high[i] > lowest_low[i]:
                log_val = np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) * np.sqrt(period)
                chop[i] = 100 * log_val / np.log10(period)
            else:
                chop[i] = 50.0  # neutral value
        return chop
    
    # Calculate indicators
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(high_1d, low_1d, 20)
    chop = calculate_chop(high_1d, low_1d, close_1d, 14)
    
    # Align 1d indicators to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in range regime (CHOP > 61.8)
        is_range = chop_aligned[i] > 61.8
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_spike = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_upper_aligned[i]
        breakout_down = close[i] < donchian_lower_aligned[i]
        
        if position == 0:
            # Long: Donchian breakout up + volume spike + range regime
            if breakout_up and volume_spike and is_range:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + volume spike + range regime
            elif breakout_down and volume_spike and is_range:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses Donchian midpoint OR regime shifts to trend (CHOP < 38.2)
            if close[i] < donchian_middle_aligned[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses Donchian midpoint OR regime shifts to trend (CHOP < 38.2)
            if close[i] > donchian_middle_aligned[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_ChopRegime"
timeframe = "12h"
leverage = 1.0