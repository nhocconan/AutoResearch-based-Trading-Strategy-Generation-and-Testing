#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and 12h ADX trend filter
# Donchian breakouts capture momentum in both bull and bear markets. Volume spike confirms
# institutional participation. 12h ADX > 25 ensures we only trade in strong trends,
# avoiding whipsaws in ranges. This combination targets 20-50 total trades over 4 years
# (5-12.5/year) to minimize fee drag while maintaining edge in BTC/ETH.

name = "4h_Donchian20_12hVolume_12hADX"
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
    
    # Calculate Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 12h data for volume and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Volume spike detection on 12h
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=24, min_periods=24).mean().values  # ~12 days
    vol_spike_12h = vol_12h > (vol_ma_12h * 2.0)
    vol_spike_4h = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # ADX trend filter on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate directional movement
    plus_dm = np.zeros_like(high_12h)
    minus_dm = np.zeros_like(high_12h)
    tr = np.zeros_like(high_12h)
    
    for i in range(1, len(high_12h)):
        plus_dm[i] = max(high_12h[i] - high_12h[i-1], 0)
        minus_dm[i] = max(low_12h[i-1] - low_12h[i], 0)
        if plus_dm[i] == minus_dm[i]:
            plus_dm[i] = 0
            minus_dm[i] = 0
        tr[i] = max(
            high_12h[i] - low_12h[i],
            abs(high_12h[i] - close_12h[i-1]),
            abs(low_12h[i] - close_12h[i-1])
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
    adx_strong_4h = align_htf_to_ltf(prices, df_12h, adx_strong)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 24)  # Ensure sufficient data for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_spike_4h[i]) or np.isnan(adx_strong_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian upper, volume spike, strong trend
            if close[i] > highest_high[i] and vol_spike_4h[i] and adx_strong_4h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian lower, volume spike, strong trend
            elif close[i] < lowest_low[i] and vol_spike_4h[i] and adx_strong_4h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian lower or trend weakens
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian upper or trend weakens
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals