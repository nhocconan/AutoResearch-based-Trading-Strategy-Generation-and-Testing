#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d ATR filter + volume confirmation
- Donchian(20) breakout identifies directional momentum
- 1d ATR > 1.5x 20-period MA ensures sufficient volatility for breakout follow-through
- Volume confirmation (> 1.8x 20-period MA) validates breakout strength
- Designed for 12h timeframe to capture medium-term swings with controlled trade frequency
- Works in bull via upside breakouts and bear via downside breakouts
- Target: 12-37 trades/year per symbol (50-150 total over 4 years) to avoid fee drag
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
    
    # Calculate Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d ATR MA for volatility regime filter
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14+20, 20)  # Donchian(20), ATR(14)+MA(20), vol MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND ATR > 1.5x ATR MA AND volume > 1.8x vol MA
            if (close[i] > highest_20[i] and 
                atr_1d_aligned[i] > 1.5 * atr_ma_1d_aligned[i] and 
                volume[i] > 1.8 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND ATR > 1.5x ATR MA AND volume > 1.8x vol MA
            elif (close[i] < lowest_20[i] and 
                  atr_1d_aligned[i] > 1.5 * atr_ma_1d_aligned[i] and 
                  volume[i] > 1.8 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to middle of Donchian channel OR ATR drops below 1.0x ATR MA
            exit_signal = False
            if position == 1:
                # Exit long when price < midpoint OR low volatility
                midpoint = (highest_20[i] + lowest_20[i]) / 2.0
                if close[i] < midpoint or atr_1d_aligned[i] < 1.0 * atr_ma_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > midpoint OR low volatility
                midpoint = (highest_20[i] + lowest_20[i]) / 2.0
                if close[i] > midpoint or atr_1d_aligned[i] < 1.0 * atr_ma_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dATR_VolumeFilter"
timeframe = "12h"
leverage = 1.0