#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_Regime
Hypothesis: 4h Donchian(20) breakout with volume spike and ADX regime filter.
Long when price breaks above Donchian(20) high + volume spike + ADX>25 (trending).
Short when price breaks below Donchian(20) low + volume spike + ADX>25.
Exit on Donchian(10) opposite breakout or ADX<20 (range regime).
Uses discrete position sizing (0.25) to minimize fee churn. Works in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # ADX for regime filter (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike: current > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Discrete size to reduce fee churn
    
    # Warmup: need Donchian(20), ADX(14), vol avg(20)
    start_idx = max(20, 14+14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or
            np.isnan(donchian_high_10[i]) or np.isnan(donchian_low_10[i]) or
            np.isnan(adx[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_20 = donchian_high_20[i]
        lower_20 = donchian_low_20[i]
        upper_10 = donchian_high_10[i]
        lower_10 = donchian_low_10[i]
        adx_val = adx[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry: Donchian(20) breakout with volume spike and trending regime (ADX>25)
            long_condition = (close_val > upper_20 and vol_spike and adx_val > 25)
            short_condition = (close_val < lower_20 and vol_spike and adx_val > 25)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: Donchian(10) breakdown OR ADX<20 (range regime)
            if close_val < lower_10 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Donchian(10) breakout OR ADX<20 (range regime)
            if close_val > upper_10 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0