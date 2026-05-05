#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and chop regime filter
# Long when price breaks above Donchian(20) high AND volume > 2.0x 20-period average AND chop > 61.8 (range regime)
# Short when price breaks below Donchian(20) low AND volume > 2.0x 20-period average AND chop > 61.8 (range regime)
# Exit when price crosses Donchian(20) midpoint OR chop < 38.2 (trend regime)
# Uses Donchian structure for breakouts, volume confirmation for conviction, chop filter to avoid whipsaws in trends.
# Timeframe: 4h, HTF: 1d for chop calculation. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Chopiness Index (CHOP) on 1d
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        n = len(close_arr)
        if n < window:
            return np.full(n, 50.0)
        
        atr_vals = np.zeros(n)
        for i in range(window-1, n):
            tr = np.max([
                high_arr[i] - low_arr[i],
                abs(high_arr[i] - close_arr[i-1]),
                abs(low_arr[i] - close_arr[i-1])
            ])
            if i == window-1:
                atr_vals[i] = np.mean(tr)
            else:
                atr_vals[i] = (atr_vals[i-1] * (window-1) + tr) / window
        
        # True range sum over window
        tr_sum = np.zeros(n)
        for i in range(window-1, n):
            if i == window-1:
                tr_sum[i] = np.sum([
                    high_arr[i-window+1:i+1] - low_arr[i-window+1:i+1],
                    np.abs(high_arr[i-window+1:i+1] - np.roll(close_arr[i-window+1:i+1], 1)),
                    np.abs(low_arr[i-window+1:i+1] - np.roll(close_arr[i-window+1:i+1], 1))
                ])
            else:
                tr_sum[i] = tr_sum[i-1] + np.max([
                    high_arr[i] - low_arr[i],
                    abs(high_arr[i] - close_arr[i-1]),
                    abs(low_arr[i] - close_arr[i-1])
                ]) - np.max([
                    high_arr[i-window] - low_arr[i-window],
                    abs(high_arr[i-window] - close_arr[i-window-1]),
                    abs(low_arr[i-window] - close_arr[i-window-1])
                ])
        
        # Avoid division by zero
        atr_safe = np.where(atr_vals == 0, 1e-10, atr_vals)
        chop = 100 * np.log10(tr_sum / window / atr_safe) / np.log10(window)
        return np.where(np.isnan(chop) | np.isinf(chop), 50.0, chop)
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate Donchian channels on 4h
    def donchian_channels(high_arr, low_arr, window=20):
        n = len(high_arr)
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        middle = np.full(n, np.nan)
        
        for i in range(window-1, n):
            upper[i] = np.max(high_arr[i-window+1:i+1])
            lower[i] = np.min(low_arr[i-window+1:i+1])
            middle[i] = (upper[i] + lower[i]) / 2
        
        return upper, lower, middle
    
    donch_upper, donch_lower, donch_middle = donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donch_upper[i]) or 
            np.isnan(donch_lower[i]) or 
            np.isnan(donch_middle[i]) or 
            np.isnan(volume_filter[i]) or 
            np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade in range markets (CHOP > 61.8)
        in_range_regime = chop_1d_aligned[i] > 61.8
        
        if position == 0 and in_range_regime:
            # Long: break above Donchian upper with volume spike
            if close[i] > donch_upper[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian lower with volume spike
            elif close[i] < donch_lower[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses Donchian middle OR chop < 38.2 (trend regime)
            if close[i] < donch_middle[i] or chop_1d_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses Donchian middle OR chop < 38.2 (trend regime)
            if close[i] > donch_middle[i] or chop_1d_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals