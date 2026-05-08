#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume spike and 1w ADX trend filter
# Breakouts above upper Donchian(20) or below lower Donchian(20) indicate strong momentum.
# 1w volume spike confirms institutional participation. 1w ADX > 25 ensures we only trade
# in strong trends, avoiding whipsaws in ranges. Works in both bull and bear markets.
# Targets 15-25 trades per year (~60-100 total over 4 years) to minimize fee drag.

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
    
    # Get 1w data for volume and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w volume spike (24-period MA ~ 12 days)
    vol_ma = pd.Series(df_1w['volume'].values).rolling(window=24, min_periods=24).mean()
    vol_spike = df_1w['volume'].values > (vol_ma.values * 2.0)
    vol_spike_1d = align_htf_to_ltf(prices, df_1w, vol_spike)
    
    # Calculate ADX(14) on 1w
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
    
    plus_di14 = np.where(tr14 != 0, 100 * (plus_dm14 / tr14), 0)
    minus_di14 = np.where(tr14 != 0, 100 * (minus_dm14 / tr14), 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilder_smooth(dx, 14)
    
    adx_strong = adx > 25
    adx_weak = adx < 20
    adx_strong_1d = align_htf_to_ltf(prices, df_1w, adx_strong)
    adx_weak_1d = align_htf_to_ltf(prices, df_1w, adx_weak)
    
    # Calculate Donchian(20) on daily
    donchian_up = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_up[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_spike_1d[i]) or 
            np.isnan(adx_strong_1d[i]) or np.isnan(adx_weak_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian up, volume spike, strong trend
            if close[i] > donchian_up[i] and vol_spike_1d[i] and adx_strong_1d[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, volume spike, strong trend
            elif close[i] < donchian_low[i] and vol_spike_1d[i] and adx_strong_1d[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian low or trend weakens
            if close[i] < donchian_low[i] or adx_weak_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian up or trend weakens
            if close[i] > donchian_up[i] or adx_weak_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals