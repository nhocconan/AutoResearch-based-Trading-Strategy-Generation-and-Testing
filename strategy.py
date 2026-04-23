#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d ATR filter and volume confirmation.
- Uses Donchian channel breakout from 20-period high/low on 6h timeframe
- 1d ATR(14) as volatility filter: only trade when ATR > 1.5x 50-period average (avoid low-vol chop)
- Volume > 1.5x 20-period average for confirmation
- Position size: 0.25 discrete level to minimize fee churn
- Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
- Works in both bull/bear via volatility filter + breakout structure
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period) on 6h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR filter: > 1.5x 50-period average (avoid low-vol chop)
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_filter = atr_14 > 1.5 * atr_ma_50
    
    # Align 1d ATR filter to 6h timeframe
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20, 50)  # Donchian, volume MA, ATR MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > high_20[i]  # Close above 20-period high
        breakout_down = close[i] < low_20[i]  # Close below 20-period low
        
        if position == 0:
            # Long: Donchian breakout up AND ATR filter AND volume confirmation
            if breakout_up and atr_filter_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND ATR filter AND volume confirmation
            elif breakout_down and atr_filter_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian breakdown (close below 20-period low) OR ATR filter fails
            if breakout_down or not atr_filter_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian breakout (close above 20-period high) OR ATR filter fails
            if breakout_up or not atr_filter_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1dATR_Filter_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0