#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume spike and 1w ADX trend filter
# Donchian channels identify breakouts from 20-day price extremes. Volume spike confirms
# institutional participation. Weekly ADX > 25 ensures we only trade in strong weekly trends,
# avoiding whipsaws in ranges. This combination works in both bull and bear markets by
# filtering for strong trends only. Targets 10-25 trades per year (~40-100 total over 4 years)
# to minimize fee drag.

name = "1d_Donchian20_1wVolume_1wADX"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian, volume, and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Donchian channels (20-period)
    upper_20 = np.full_like(high_1w, np.nan)
    lower_20 = np.full_like(low_1w, np.nan)
    for i in range(20, len(high_1w)):
        upper_20[i] = np.max(high_1w[i-20:i])
        lower_20[i] = np.min(low_1w[i-20:i])
    
    # Align Donchian levels to daily timeframe (use previous week's levels)
    upper_20_d = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_d = align_htf_to_ltf(prices, df_1w, lower_20)
    
    # Volume spike detection on 1w (volume > 2x 20-period average)
    vol_ma_20 = np.full_like(volume_1w, np.nan)
    for i in range(20, len(volume_1w)):
        vol_ma_20[i] = np.mean(volume_1w[i-20:i])
    vol_spike_1w = volume_1w > (vol_ma_20 * 2.0)
    vol_spike_d = align_htf_to_ltf(prices, df_1w, vol_spike_1w)
    
    # ADX trend filter on 1w
    plus_dm = np.zeros_like(high_1w)
    minus_dm = np.zeros_like(high_1w)
    tr = np.zeros_like(high_1w)
    
    for i in range(1, len(high_1w)):
        plus_dm[i] = max(high_1w[i] - high_1w[i-1], 0)
        minus_dm[i] = max(low_1w[i-1] - low_1w[i], 0)
        if plus_dm[i] == minus_dm[i]:
            plus_dm[i] = 0
            minus_dm[i] = 0
        tr[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    
    # Wilder smoothing
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr20 = wilder_smooth(tr, 20)
    plus_dm20 = wilder_smooth(plus_dm, 20)
    minus_dm20 = wilder_smooth(minus_dm, 20)
    
    # Avoid division by zero
    plus_di20 = np.where(tr20 != 0, 100 * (plus_dm20 / tr20), 0)
    minus_di20 = np.where(tr20 != 0, 100 * (minus_dm20 / tr20), 0)
    
    dx = np.where((plus_di20 + minus_di20) != 0, 
                  100 * np.abs(plus_di20 - minus_di20) / (plus_di20 + minus_di20), 0)
    adx = wilder_smooth(dx, 20)
    
    adx_strong = adx > 25
    adx_weak = adx < 20
    adx_strong_d = align_htf_to_ltf(prices, df_1w, adx_strong)
    adx_weak_d = align_htf_to_ltf(prices, df_1w, adx_weak)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_20_d[i]) or np.isnan(lower_20_d[i]) or 
            np.isnan(vol_spike_d[i]) or 
            np.isnan(adx_strong_d[i]) or np.isnan(adx_weak_d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper Donchian, volume spike, strong trend
            if close[i] > upper_20_d[i] and vol_spike_d[i] and adx_strong_d[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian, volume spike, strong trend
            elif close[i] < lower_20_d[i] and vol_spike_d[i] and adx_strong_d[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to lower Donchian or trend weakens
            if close[i] < lower_20_d[i] or adx_weak_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to upper Donchian or trend weakens
            if close[i] > upper_20_d[i] or adx_weak_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals