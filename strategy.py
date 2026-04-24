#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 1d ATR filter and volume confirmation.
- Long when price breaks above Donchian(20) high AND 1d ATR(14) > 1.3 * 1d ATR(50) (expanding volatility regime)
- Short when price breaks below Donchian(20) low AND 1d ATR(14) > 1.3 * 1d ATR(50) (expanding volatility regime)
- Volume confirmation: current volume > 1.5 * 20-period average volume
- Exit on opposite Donchian breakout (low for long exit, high for short exit)
- Uses 4h primary with 1d HTF to target 75-200 trades over 4 years (19-50/year)
- Donchian provides adaptive structure; ATR filter captures volatility expansion; volume avoids fakeouts
- Works in both bull (breakouts with volume) and bear (breakdowns with volume) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels using previous 20 bars (no look-ahead)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Get 1d data ONCE before loop for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR(50)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # ATR filter: expanding volatility regime (ATR(14) > 1.3 * ATR(50))
    atr_expanding = atr_14 > (1.3 * atr_50)
    
    # Align 1d ATR filter to 4h timeframe (waits for completed 1d bar)
    atr_expanding_aligned = align_htf_to_ltf(prices, df_1d, atr_expanding)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need Donchian(20), volume MA, and ATR(50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(atr_expanding_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high AND expanding volatility AND volume confirmation
            if close[i] > highest_20[i] and atr_expanding_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low AND expanding volatility AND volume confirmation
            elif close[i] < lowest_20[i] and atr_expanding_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below Donchian low
            if close[i] < lowest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above Donchian high
            if close[i] > highest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATRExpanding_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0