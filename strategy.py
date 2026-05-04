#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + ADX regime filter
# Uses Donchian channel breakouts for trend capture, confirmed by 1d volume spikes
# and filtered by ADX > 25 to avoid ranging markets. Designed for 20-50 trades/year
# (~80-200 total over 4 years) to minimize fee drag. Works in bull/bear markets
# by using ADX regime to only trade when trending conditions exist.

name = "4h_Donchian20_1dVolumeSpike_ADXRegime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume spike and ADX - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ADX(14)
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(low_1d)
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(0, high_1d[i] - high_1d[i-1])
        minus_dm[i] = max(0, low_1d[i-1] - low_1d[i])
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])],
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period average)
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma)
    
    # Align 1d indicators to 4h timeframe (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when ADX > 25 (trending market)
        if adx_aligned[i] <= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: require volume spike on 1d
        if volume_spike_aligned[i] < 0.5:  # no spike
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper channel
            if close[i] > highest_high[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower channel
            elif close[i] < lowest_low[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches Donchian middle or breaks lower channel
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < donchian_mid or close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches Donchian middle or breaks upper channel
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > donchian_mid or close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals