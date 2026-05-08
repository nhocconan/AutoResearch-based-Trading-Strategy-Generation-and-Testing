#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and 1w ADX trend filter
# Daily breakouts above/below 20-day Donchian channels capture strong trends.
# Weekly volume surge confirms institutional participation. Weekly ADX > 25 ensures
# we trade only in strong trends, avoiding whipsaws in ranges. This works in both
# bull and bear markets by filtering for strong trends only.
# Targets 7-25 trades per year (30-100 total over 4 years) to minimize fee drag.

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
    
    # Donchian channels (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.max(arr[i - window + 1:i + 1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.min(arr[i - window + 1:i + 1])
        return res
    
    upper = rolling_max(high_1w, 20)
    lower = rolling_min(low_1w, 20)
    
    # Volume spike detection on weekly
    vol_ma = np.full_like(volume_1w, np.nan)
    for i in range(20, len(volume_1w)):  # 20-week MA
        vol_ma[i] = np.mean(volume_1w[i - 20:i])
    vol_spike = volume_1w > (vol_ma * 2.0)
    
    # ADX trend filter on weekly (14-period)
    plus_dm = np.zeros_like(high_1w)
    minus_dm = np.zeros_like(high_1w)
    tr = np.zeros_like(high_1w)
    
    for i in range(1, len(high_1w)):
        plus_dm[i] = max(high_1w[i] - high_1w[i - 1], 0)
        minus_dm[i] = max(low_1w[i - 1] - low_1w[i], 0)
        if plus_dm[i] == minus_dm[i]:
            plus_dm[i] = 0
            minus_dm[i] = 0
        tr[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i - 1]),
            abs(low_1w[i] - close_1w[i - 1])
        )
    
    def wilder_smooth(arr, period):
        res = np.full_like(arr, np.nan)
        if len(arr) < period:
            return res
        res[period - 1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            res[i] = res[i - 1] - (res[i - 1] / period) + arr[i]
        return res
    
    tr14 = wilder_smooth(tr, 14)
    plus_dm14 = wilder_smooth(plus_dm, 14)
    minus_dm14 = wilder_smooth(minus_dm, 14)
    
    plus_di14 = np.where(tr14 != 0, 100 * (plus_dm14 / tr14), 0)
    minus_di14 = np.where(tr14 != 0, 100 * (minus_dm14 / tr14), 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0,
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilder_smooth(dx, 14)
    
    adx_strong = adx > 25
    adx_weak = adx < 20
    
    # Align weekly indicators to daily timeframe
    upper_d = align_htf_to_ltf(prices, df_1w, upper)
    lower_d = align_htf_to_ltf(prices, df_1w, lower)
    vol_spike_d = align_htf_to_ltf(prices, df_1w, vol_spike)
    adx_strong_d = align_htf_to_ltf(prices, df_1w, adx_strong)
    adx_weak_d = align_htf_to_ltf(prices, df_1w, adx_weak)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure sufficient data for Donchian (20) + vol MA (20)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_d[i]) or np.isnan(lower_d[i]) or 
            np.isnan(vol_spike_d[i]) or np.isnan(adx_strong_d[i]) or 
            np.isnan(adx_weak_d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper Donchian, volume spike, strong trend
            if close[i] > upper_d[i] and vol_spike_d[i] and adx_strong_d[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian, volume spike, strong trend
            elif close[i] < lower_d[i] and vol_spike_d[i] and adx_strong_d[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to lower Donchian or trend weakens
            if close[i] < lower_d[i] or adx_weak_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to upper Donchian or trend weakens
            if close[i] > upper_d[i] or adx_weak_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals