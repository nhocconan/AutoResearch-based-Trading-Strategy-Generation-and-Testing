#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_ATRRegime
Hypothesis: On 4h timeframe, enter long when price breaks above 20-period Donchian high AND volume > 2.0x 20-period average AND ATR(14) is in normal range (not too low/high). Enter short when price breaks below 20-period Donchian low AND volume spike AND normal ATR. Uses ATR-based regime filter to avoid choppy markets and extreme volatility periods. Designed for moderate trade frequency (20-50/year) with strong edge in both bull and bear markets via volatility-adjusted breakouts.
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
    
    # Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    # ATR for volatility regime filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR regime: avoid extremes (too low = chop, too high = panic)
    atr_mean = pd.Series(atr).rolling(window=50, min_periods=20).mean().values
    atr_ratio = atr / np.maximum(atr_mean, 1e-10)
    volatility_normal = (atr_ratio > 0.5) & (atr_ratio < 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20), volume MA (20), ATR (14), ATR mean (50)
    start_idx = max(lookback, 20, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or np.isnan(atr_mean[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        if position == 0:
            # Long: breakout up + volume spike + normal volatility
            long_signal = breakout_up and volume_spike[i] and volatility_normal[i]
            
            # Short: breakout down + volume spike + normal volatility
            short_signal = breakout_down and volume_spike[i] and volatility_normal[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below Donchian low OR volatility extreme
            if close[i] < lowest_low[i] or not volatility_normal[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above Donchian high OR volatility extreme
            if close[i] > highest_high[i] or not volatility_normal[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ATRRegime"
timeframe = "4h"
leverage = 1.0