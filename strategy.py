#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + volume spike + 1d choppiness regime filter
    # Long: price breaks above Donchian(20) high AND volume > 1.5 * avg_volume(20) AND 1d CHOP > 61.8 (range)
    # Short: price breaks below Donchian(20) low AND volume > 1.5 * avg_volume(20) AND 1d CHOP > 61.8 (range)
    # Exit: price reverts to Donchian(20) midline OR CHOP < 38.2 (trend)
    # Using 4h for price action/volume, 1d for chop regime (range detection)
    # Discrete position sizing (0.25) to minimize fee drag
    # Target: 20-50 trades/year (~80-200 over 4 years) to avoid overtrading
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 4h volume average (20-period)
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    avg_volume = rolling_mean(volume, 20)
    volume_spike = volume > (1.5 * avg_volume)
    
    # Get 1d data for choppiness index (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d choppiness index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Sum of TR over 14 periods
    def rolling_sum(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.sum(arr[i - window + 1:i + 1])
        return result
    
    sum_tr_14 = rolling_sum(tr, 14)
    
    # Highest high and lowest low over 14 periods
    hh_14 = rolling_max(high_1d, 14)
    ll_14 = rolling_min(low_1d, 14)
    
    # Choppiness Index = 100 * log10(sum_tr_14 / (hh_14 - ll_14)) / log10(14)
    chop_denominator = hh_14 - ll_14
    chop_ratio = np.where(chop_denominator > 0, sum_tr_14 / chop_denominator, np.nan)
    chop_1d = 100 * np.log10(chop_ratio) / np.log10(14)
    
    # Align 1d chop to 4h (wait for completed 1d bar)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when 1d CHOP > 61.8 (range-bound market)
        in_range = chop_1d_aligned[i] > 61.8
        # Exit regime: when market starts trending (CHOP < 38.2)
        trending = chop_1d_aligned[i] < 38.2
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high[i-1]  # break above previous period's high
        breakout_down = close[i] < donchian_low[i-1]  # break below previous period's low
        
        # Entry logic: Donchian breakout + volume spike + range regime
        long_entry = breakout_up and volume_spike[i] and in_range
        short_entry = breakout_down and volume_spike[i] and in_range
        
        # Exit logic: price reverts to midline OR regime shifts to trending
        long_exit = (close[i] < donchian_mid[i]) or trending
        short_exit = (close[i] > donchian_mid[i]) or trending
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0