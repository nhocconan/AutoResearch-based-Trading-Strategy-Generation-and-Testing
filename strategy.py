#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume spike and 1w ADX trend filter
# Donchian(20) identifies key support/resistance levels. Breakouts above upper band or below lower band
# indicate strong momentum. Volume spike confirms institutional participation. 1w ADX > 25 ensures we only
# trade in strong trends, avoiding whipsaws in ranges. This combination works in both bull and bear markets
# by filtering for strong trends only. Targets 15-25 trades per year (~60-100 total over 4 years) to minimize fee drag.

name = "12h_Donchian20_1dVolume_1wADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(lookback-1, n):
        upper[i] = np.max(high[i-lookback+1:i+1])
        lower[i] = np.min(low[i-lookback+1:i+1])
    
    # Get 1d data for volume spike detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Volume spike detection on 1d (20-day average)
    vol_ma = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean()
    vol_spike = df_1d['volume'].values > (vol_ma.values * 2.0)
    vol_spike_12h = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate ADX(14) on weekly
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
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
    adx_strong_12h = align_htf_to_ltf(prices, df_1w, adx_strong)
    adx_weak_12h = align_htf_to_ltf(prices, df_1w, adx_weak)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = lookback-1  # Ensure sufficient data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_spike_12h[i]) or 
            np.isnan(adx_strong_12h[i]) or np.isnan(adx_weak_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper Donchian band, volume spike, strong trend
            if close[i] > upper[i] and vol_spike_12h[i] and adx_strong_12h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian band, volume spike, strong trend
            elif close[i] < lower[i] and vol_spike_12h[i] and adx_strong_12h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to lower band or trend weakens
            if close[i] < lower[i] or adx_weak_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to upper band or trend weakens
            if close[i] > upper[i] or adx_weak_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals