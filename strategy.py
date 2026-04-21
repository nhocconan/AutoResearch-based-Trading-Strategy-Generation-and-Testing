#!/usr/bin/env python3
"""
4h_Donchian20_VolumeSpike_SqueezeBreakout
Hypothesis: On 4h timeframe, breakouts above Donchian(20) high or below Donchian(20) low with volume spikes and volatility squeeze (low ATR) capture high-probability moves. Works in bull/bear by requiring volume confirmation and squeeze filter to avoid whipsaws. Target: 20-50 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 1d data for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First TR is undefined
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Donchian(20) on 4h
    high_20 = pd.Series(prices['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(prices['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike = prices['volume'].values > (2.0 * vol_ma)
    
    # Volatility squeeze: daily ATR below its 50-period median (low volatility regime)
    atr_median = pd.Series(atr_1d_aligned).rolling(window=50, min_periods=50).median().values
    volatility_squeeze = atr_1d_aligned < atr_median
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(atr_median[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol_spike = volume_spike[i]
        squeeze = volatility_squeeze[i]
        
        if position == 0:
            # Long: breakout above Donchian high + volume spike + volatility squeeze
            if price > high_20[i] and vol_spike and squeeze:
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low + volume spike + volatility squeeze
            elif price < low_20[i] and vol_spike and squeeze:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price retouches Donchian mid-point or volatility expands
            mid_point = (high_20[i] + low_20[i]) / 2
            if price < mid_point or not squeeze:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price retouches Donchian mid-point or volatility expands
            mid_point = (high_20[i] + low_20[i]) / 2
            if price > mid_point or not squeeze:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_SqueezeBreakout"
timeframe = "4h"
leverage = 1.0