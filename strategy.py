#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation.
- Donchian(20) breakouts capture momentum in both bull and bear markets via clear price channels.
- 1d ATR(14) normalized by price acts as a volatility regime filter: trade only when ATR/price > 0.02 (high volatility environments where breakouts are more reliable).
- Volume confirmation (>1.5x 20-period average) reduces false breakouts in low-volume periods.
- Discrete position sizing (0.25) minimizes fee churn while allowing meaningful returns.
- Target trades: 75-200 total over 4 years (19-50/year) to avoid fee drag on 4h timeframe.
- Uses close-based exits to respect engine semantics and avoid look-ahead.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d ATR(14) as percentage of price (volatility filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low (no previous close)
    tr[0] = tr1[0]
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_percent_1d = atr_14_1d / close_1d  # ATR as fraction of price
    atr_percent_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_percent_1d)
    
    # Donchian(20) channels on 4h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(atr_percent_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian upper band with volume confirmation and high volatility regime
            if close[i] > highest_20[i] and volume_confirm[i] and atr_percent_1d_aligned[i] > 0.02:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian lower band with volume confirmation and high volatility regime
            elif close[i] < lowest_20[i] and volume_confirm[i] and atr_percent_1d_aligned[i] > 0.02:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Donchian lower band (breakdown) OR volatility drops
            if close[i] < lowest_20[i] or atr_percent_1d_aligned[i] < 0.015:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian upper band (breakout) OR volatility drops
            if close[i] > highest_20[i] or atr_percent_1d_aligned[i] < 0.015:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ATRVolFilter_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0