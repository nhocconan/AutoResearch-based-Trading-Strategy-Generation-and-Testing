#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 1d volume spike + ADX trend filter
# Donchian(20) breakout captures breakouts in trending markets
# 1d volume spike confirms institutional interest
# ADX > 25 filters for trending markets only, avoiding whipsaws in ranges
# Designed for low trade frequency (target 20-40/year) with clear trend following
# Works in bull markets (breakouts up) and bear markets (breakouts down)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d volume spike: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma)
    
    # 4h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h ADX(14) for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = np.maximum(high[1:], low[:-1]) - np.minimum(high[1:], low[:-1])
    tr2 = np.abs(high[1:] - low[:-1])
    tr3 = np.abs(low[1:] - high[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 4h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(adx[i]) or np.isnan(volume_spike_aligned[i])):
            continue
        
        # Only trade in trending markets (ADX > 25)
        if adx[i] > 25:
            # Long breakout: price breaks above Donchian high with volume spike
            if close[i] > high_max[i] and volume_spike_aligned[i] and position <= 0:
                position = 1
                signals[i] = position_size
            # Short breakdown: price breaks below Donchian low with volume spike
            elif close[i] < low_min[i] and volume_spike_aligned[i] and position >= 0:
                position = -1
                signals[i] = -position_size
            # Exit when price returns to mid-channel
            elif position == 1 and close[i] < (high_max[i] + low_min[i]) / 2:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > (high_max[i] + low_min[i]) / 2:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian_VolumeSpike_ADX"
timeframe = "4h"
leverage = 1.0