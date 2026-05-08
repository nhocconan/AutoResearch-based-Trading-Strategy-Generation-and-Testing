#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1d ADX trend filter
# Breakouts above/below 20-period Donchian channels capture strong momentum moves.
# 1d volume > 1.5x 20-period average confirms institutional participation.
# 1d ADX > 25 ensures we only trade in strong trends, avoiding whipsaws in ranges.
# This combination works in both bull and bear markets by filtering for strong trends only.
# Targets 20-50 trades per year (~80-200 total over 4 years) to minimize fee drag.

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
    
    # Get 1d data for Donchian, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period high/low)
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume MA (20-period)
    vol_ma = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX(14) on 1d
    plus_dm = np.zeros_like(high_20)
    minus_dm = np.zeros_like(high_20)
    tr = np.zeros_like(high_20)
    
    for i in range(1, len(high_20)):
        plus_dm[i] = max(high_20[i] - high_20[i-1], 0)
        minus_dm[i] = max(low_20[i-1] - low_20[i], 0)
        if plus_dm[i] == minus_dm[i]:
            plus_dm[i] = 0
            minus_dm[i] = 0
        tr[i] = max(
            high_20[i] - low_20[i],
            abs(high_20[i] - df_1d['close'].iloc[i-1]),
            abs(low_20[i] - df_1d['close'].iloc[i-1])
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
    
    # Align indicators to 4h timeframe
    high_20_4h = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_4h = align_htf_to_ltf(prices, df_1d, low_20)
    vol_ma_4h = align_htf_to_ltf(prices, df_1d, vol_ma)
    adx_4h = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20_4h[i]) or np.isnan(low_20_4h[i]) or 
            np.isnan(vol_ma_4h[i]) or np.isnan(adx_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, volume > 1.5x MA, strong trend
            if close[i] > high_20_4h[i] and volume[i] > (vol_ma_4h[i] * 1.5) and adx_4h[i] > 25:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, volume > 1.5x MA, strong trend
            elif close[i] < low_20_4h[i] and volume[i] > (vol_ma_4h[i] * 1.5) and adx_4h[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian low or trend weakens
            if close[i] < low_20_4h[i] or adx_4h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian high or trend weakens
            if close[i] > high_20_4h[i] or adx_4h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals