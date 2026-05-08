#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Fibonacci pivot breakout with 12h volume confirmation and 1d trend filter
# Fibonacci pivots identify key support/resistance levels. Breakouts above R2 or below S2
# indicate strong momentum. Volume spike on 12h confirms institutional participation.
# 1d ADX > 25 ensures we only trade in strong trends, avoiding whipsaws in ranges.
# This combination works in both bull and bear markets by filtering for strong trends only.
# Targets 20-40 trades per year (~80-160 total over 4 years) to minimize fee drag.

name = "6h_FibPivot_R2S2_12hVolume_1dADX"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data for Fibonacci pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Fibonacci pivots from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Fibonacci pivot levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    R1 = pivot + 0.382 * range_hl
    S1 = pivot - 0.382 * range_hl
    R2 = pivot + 0.618 * range_hl
    S2 = pivot - 0.618 * range_hl
    R3 = pivot + 1.0 * range_hl
    S3 = pivot - 1.0 * range_hl
    
    # Align Fibonacci levels to 6h timeframe (use previous day's levels)
    R2_6h = align_htf_to_ltf(prices, df_1d, R2)
    S2_6h = align_htf_to_ltf(prices, df_1d, S2)
    
    # Volume spike detection on 12h
    vol_ma = pd.Series(df_12h['volume'].values).rolling(window=10, min_periods=10).mean()  # ~5 days
    vol_spike_12h = df_12h['volume'].values > (vol_ma.values * 2.0)
    vol_spike_6h = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # ADX trend filter on 1d
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
    adx_strong_6h = align_htf_to_ltf(prices, df_1d, adx_strong)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 10  # Ensure sufficient data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R2_6h[i]) or np.isnan(S2_6h[i]) or 
            np.isnan(vol_spike_6h[i]) or np.isnan(adx_strong_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R2, volume spike, strong trend
            if close[i] > R2_6h[i] and vol_spike_6h[i] and adx_strong_6h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S2, volume spike, strong trend
            elif close[i] < S2_6h[i] and vol_spike_6h[i] and adx_strong_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to S2 or trend weakens
            if close[i] < S2_6h[i] or not adx_strong_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to R2 or trend weakens
            if close[i] > R2_6h[i] or not adx_strong_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals