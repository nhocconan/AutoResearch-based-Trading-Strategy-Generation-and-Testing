#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d volume spike and 1d ADX trend filter
# Williams %R identifies overbought/oversold conditions. Readings below -80 indicate oversold,
# above -20 indicate overbought. Combined with volume spikes (institutional participation) and
# ADX > 25 (strong trend), this captures mean-reversion bounces within strong trends.
# Works in both bull and bear markets by trading pullbacks in trending markets.
# Targets 12-37 trades per year (~50-150 total over 4 years) to minimize fee drag.

name = "12h_WilliamsR_1dVolume_1dADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams %R(14) on daily
    highest_high = np.full_like(high_1d, np.nan)
    lowest_low = np.full_like(low_1d, np.nan)
    for i in range(13, len(high_1d)):
        highest_high[i] = np.max(high_1d[i-13:i+1])
        lowest_low[i] = np.min(low_1d[i-13:i+1])
    
    williams_r = np.where((highest_high - lowest_low) != 0,
                          -100 * (highest_high - close_1d) / (highest_high - lowest_low),
                          -50)
    
    # Williams %R signals: oversold < -80, overbought > -20
    williams_oversold = williams_r < -80
    williams_overbought = williams_r > -20
    
    # Volume spike detection on 1d
    vol_ma = np.full_like(volume_1d, np.nan)
    for i in range(14, len(volume_1d)):
        vol_ma[i] = np.mean(volume_1d[i-14:i+1])
    vol_spike = volume_1d > (vol_ma * 2.0)
    
    # ADX trend filter on 1d (14-period)
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
    
    plus_di14 = np.where(tr14 != 0, 100 * (plus_dm14 / tr14), 0)
    minus_di14 = np.where(tr14 != 0, 100 * (minus_dm14 / tr14), 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilder_smooth(dx, 14)
    
    adx_strong = adx > 25
    adx_weak = adx < 20
    
    # Align all 1d indicators to 12h timeframe (use previous day's values)
    williams_oversold_12h = align_htf_to_ltf(prices, df_1d, williams_oversold)
    williams_overbought_12h = align_htf_to_ltf(prices, df_1d, williams_overbought)
    vol_spike_12h = align_htf_to_ltf(prices, df_1d, vol_spike)
    adx_strong_12h = align_htf_to_ltf(prices, df_1d, adx_strong)
    adx_weak_12h = align_htf_to_ltf(prices, df_1d, adx_weak)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 28  # Ensure sufficient data for Williams %R and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_oversold_12h[i]) or np.isnan(williams_overbought_12h[i]) or 
            np.isnan(vol_spike_12h[i]) or np.isnan(adx_strong_12h[i]) or np.isnan(adx_weak_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R oversold, volume spike, strong trend
            if williams_oversold_12h[i] and vol_spike_12h[i] and adx_strong_12h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought, volume spike, strong trend
            elif williams_overbought_12h[i] and vol_spike_12h[i] and adx_strong_12h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns above -50 or trend weakens
            if williams_overbought_12h[i] or adx_weak_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns below -50 or trend weakens
            if williams_oversold_12h[i] or adx_weak_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals