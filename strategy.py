#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 12h volume spike + 1d ADX trend filter
# Donchian breakouts capture momentum in trending markets. Volume spike confirms institutional participation.
# 1d ADX > 25 ensures we only trade in strong trends, avoiding whipsaws in ranges.
# Exits occur when price returns to the Donchian midpoint or trend weakens (ADX < 20).
# Targets 12-37 trades per year (~50-150 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by filtering for strong trends only.

name = "12h_Donchian20_12hVolume_1dADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels on 12h
    lookback = 20
    dc_high = np.full_like(high, np.nan)
    dc_low = np.full_like(low, np.nan)
    dc_mid = np.full_like(close, np.nan)
    
    for i in range(lookback, n):
        dc_high[i] = np.max(high[i-lookback:i])
        dc_low[i] = np.min(low[i-lookback:i])
        dc_mid[i] = (dc_high[i] + dc_low[i]) / 2.0
    
    # 12h volume spike
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (vol_ma.values * 2.0)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    tr = np.zeros_like(high_1d)
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(high_1d[i] - high_1d[i-1], 0)
        minus_dm[i] = max(low_1d[i-1] - low_1d[i], 0)
        if plus_dm[i] == minus_dm[i]:
            plus_dm[i] = 0
            minus_dm[i] = 0
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    # Smooth TR, +DM, -DM
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr14 = wilder_smooth(tr, 14)
    plus_dm14 = wilder_smooth(plus_dm, 14)
    minus_dm14 = wilder_smooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di14 = np.where(tr14 != 0, 100 * (plus_dm14 / tr14), 0)
    minus_di14 = np.where(tr14 != 0, 100 * (minus_dm14 / tr14), 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilder_smooth(dx, 14)
    
    adx_strong = adx > 25
    adx_weak = adx < 20
    adx_strong_aligned = align_htf_to_ltf(prices, df_1d, adx_strong)
    adx_weak_aligned = align_htf_to_ltf(prices, df_1d, adx_weak)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 40)  # Ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or np.isnan(dc_mid[i]) or 
            np.isnan(vol_spike[i]) or np.isnan(adx_strong_aligned[i]) or 
            np.isnan(adx_weak_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, volume spike, strong trend
            if close[i] > dc_high[i] and vol_spike[i] and adx_strong_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, volume spike, strong trend
            elif close[i] < dc_low[i] and vol_spike[i] and adx_strong_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint or trend weakens
            if close[i] < dc_mid[i] or adx_weak_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint or trend weakens
            if close[i] > dc_mid[i] or adx_weak_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals