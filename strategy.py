#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and 1w ADX trend filter
# Donchian breakouts capture strong momentum moves. Volume confirmation ensures institutional
# participation. 1w ADX > 25 filters for strong trends only, avoiding whipsaws in ranges.
# This strategy works in both bull and bear markets by trading strong trends only.
# Targets 20-30 trades per year (~80-120 total over 4 years) to minimize fee drag.

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
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Donchian channels (20-period) on 1w
    donchian_high = np.full_like(high_1w, np.nan)
    donchian_low = np.full_like(low_1w, np.nan)
    
    for i in range(20, len(high_1w)):
        donchian_high[i] = np.max(high_1w[i-20:i])
        donchian_low[i] = np.min(low_1w[i-20:i])
    
    # Volume spike detection on 1w (current volume > 2x 20-period average)
    vol_ma = np.full_like(volume_1w, np.nan)
    for i in range(20, len(volume_1w)):
        vol_ma[i] = np.mean(volume_1w[i-20:i])
    vol_spike = volume_1w > (vol_ma * 2.0)
    
    # ADX trend filter on 1w (ADX > 25 indicates strong trend)
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
    
    # Wilder smoothing function
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
    
    # Align all 1w indicators to 1d timeframe
    donchian_high_1d = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_1d = align_htf_to_ltf(prices, df_1w, donchian_low)
    vol_spike_1d = align_htf_to_ltf(prices, df_1w, vol_spike)
    adx_strong_1d = align_htf_to_ltf(prices, df_1w, adx_strong)
    adx_weak_1d = align_htf_to_ltf(prices, df_1w, adx_weak)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_1d[i]) or np.isnan(donchian_low_1d[i]) or 
            np.isnan(vol_spike_1d[i]) or np.isnan(adx_strong_1d[i]) or 
            np.isnan(adx_weak_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, volume spike, strong trend
            if close[i] > donchian_high_1d[i] and vol_spike_1d[i] and adx_strong_1d[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, volume spike, strong trend
            elif close[i] < donchian_low_1d[i] and vol_spike_1d[i] and adx_strong_1d[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian low or trend weakens
            if close[i] < donchian_low_1d[i] or adx_weak_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian high or trend weakens
            if close[i] > donchian_high_1d[i] or adx_weak_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals