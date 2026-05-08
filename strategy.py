#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1d ADX trend filter
# Breakouts above/below 20-period high/low capture strong momentum. Volume surge confirms
# institutional participation. ADX > 25 ensures trading only in strong trends, avoiding
# whipsaws in ranges. Works in bull/bear by filtering for strong trends only.
# Targets ~25 trades/year (~100 total over 4 years) to minimize fee drag.

name = "4h_Donchian20_1dVolume_1dADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian channels (20-period) on 4h close
    high_max = pd.Series(close).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    # 1d volume spike detection (volume > 2x 20-day average)
    vol_ma = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_spike_raw = df_1d['volume'].values > (vol_ma * 2.0)
    vol_spike = align_htf_to_ltf(prices, df_1d, vol_spike_raw)
    
    # 1d ADX calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    
    plus_di14 = np.where(tr14 != 0, 100 * (plus_dm14 / tr14), 0)
    minus_di14 = np.where(tr14 != 0, 100 * (minus_dm14 / tr14), 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilder_smooth(dx, 14)
    adx_strong = adx > 25
    adx_strong_4h = align_htf_to_ltf(prices, df_1d, adx_strong)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(vol_spike[i]) or np.isnan(adx_strong_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, volume spike, strong trend
            if close[i] > high_max[i] and vol_spike[i] and adx_strong_4h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, volume spike, strong trend
            elif close[i] < low_min[i] and vol_spike[i] and adx_strong_4h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian low or trend weakens
            if close[i] < low_min[i] or not adx_strong_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian high or trend weakens
            if close[i] > high_max[i] or not adx_strong_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals