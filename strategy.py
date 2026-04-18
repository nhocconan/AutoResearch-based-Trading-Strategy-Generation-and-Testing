#!/usr/bin/env python3
"""
4h_ADX_Filtered_Donchian_Breakout_Volume
Hypothesis: Price breaks above/below Donchian(20) with volume spike and ADX(14) > 20 trend filter.
Captures breakouts with trend strength confirmation to reduce false signals in both bull and bear markets.
Target: 20-40 trades/year to minimize fee drift while capturing strong directional moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ADX(14) for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 14, 14)  # Warmup for Donchian and ADX
    
    for i in range(start_idx, n):
        if (np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or
            np.isnan(adx[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_max[i]
        lower = low_min[i]
        adx_val = adx[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above upper band with volume spike and ADX > 20
            if price > upper and vol_spike and adx_val > 20:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with volume spike and ADX > 20
            elif price < lower and vol_spike and adx_val > 20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price closes below lower band OR ADX drops below 15 (trend weakening)
            if price < lower:
                signals[i] = 0.0
                position = 0
            elif adx_val < 15:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price closes above upper band OR ADX drops below 15 (trend weakening)
            if price > upper:
                signals[i] = 0.0
                position = 0
            elif adx_val < 15:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_ADX_Filtered_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0