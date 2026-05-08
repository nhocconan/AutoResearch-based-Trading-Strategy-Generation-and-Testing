#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume spike and 12h ADX trend filter
# Breakouts above/below 20-period high/low indicate strong momentum. 12h volume confirms participation.
# 12h ADX > 25 ensures trading only in strong trends. This filters whipsaws in sideways markets.
# Targets 20-50 trades per year (~80-200 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by requiring strong trend confirmation.

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
    
    # Get 12h data for volume spike and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h volume spike detection
    vol_ma = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean()
    vol_spike_12h = df_12h['volume'].values > (vol_ma.values * 2.0)
    vol_spike = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # ADX(14) calculation on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
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
    
    plus_di14 = np.where(tr14 != 0, 100 * (plus_dm14 / tr14), 0)
    minus_di14 = np.where(tr14 != 0, 100 * (minus_dm14 / tr14), 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilder_smooth(dx, 14)
    
    adx_strong = adx > 25
    adx_weak = adx < 20
    adx_strong_4h = align_htf_to_ltf(prices, df_12h, adx_strong)
    adx_weak_4h = align_htf_to_ltf(prices, df_12h, adx_weak)
    
    # Donchian(20) on 4h
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(lookback, len(high)):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # Align Donchian levels (use previous bar's values to avoid look-ahead)
    highest_high_aligned = np.roll(highest_high, 1)
    lowest_low_aligned = np.roll(lowest_low, 1)
    highest_high_aligned[0] = np.nan
    lowest_low_aligned[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback + 1, 20)  # Ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or 
            np.isnan(vol_spike[i]) or np.isnan(adx_strong_4h[i]) or np.isnan(adx_weak_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 20-period high, volume spike, strong trend
            if close[i] > highest_high_aligned[i] and vol_spike[i] and adx_strong_4h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 20-period low, volume spike, strong trend
            elif close[i] < lowest_low_aligned[i] and vol_spike[i] and adx_strong_4h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 20-period low or trend weakens
            if close[i] < lowest_low_aligned[i] or adx_weak_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 20-period high or trend weakens
            if close[i] > highest_high_aligned[i] or adx_weak_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals