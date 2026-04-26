#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_ATRRegime
Hypothesis: On 4h timeframe, enter long when price breaks above 20-period Donchian high AND volume > 1.8x 20-period average AND ATR(14) < ATR(50) (low volatility regime). Enter short when price breaks below 20-period Donchian low AND volume spike AND ATR(14) < ATR(50). Uses ATR ratio as regime filter to avoid whipsaws in high volatility. Designed for 20-50 trades/year with strong edge in both bull and bear markets via volatility-based regime filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.8
    
    # ATR for volatility regime filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / np.maximum(atr_50, 1e-10)
    low_volatility = atr_ratio < 1.0  # ATR(14) < ATR(50) indicates low volatility regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian warmup (20), volume MA warmup (20), ATR warmup (50)
    start_idx = max(lookback, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr_ratio[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_high = close[i] > highest_high[i]
        breakout_low = close[i] < lowest_low[i]
        
        if position == 0:
            # Long: break above Donchian high + volume spike + low volatility regime
            long_signal = breakout_high and volume_spike[i] and low_volatility[i]
            
            # Short: break below Donchian low + volume spike + low volatility regime
            short_signal = breakout_low and volume_spike[i] and low_volatility[i]
            
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
            # Exit: price breaks below Donchian low OR volatility expands (stoploss proxy)
            if close[i] < lowest_low[i] or not low_volatility[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above Donchian high OR volatility expands
            if close[i] > highest_high[i] or not low_volatility[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ATRRegime"
timeframe = "4h"
leverage = 1.0