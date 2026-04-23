#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 Breakout with 1d ATR Trend Filter and Volume Spike
- Uses tight entry conditions (Camarilla R1/S1 breakout + 1d ATR-based trend + volume > 1.5x 20-period MA)
- Designed for 12h timeframe to balance trade frequency and noise reduction
- Target: 12-37 trades/year per symbol (50-150 total over 4 years) to avoid fee drag
- Works in both bull and bear markets via ATR-based trend filter (strong moves) and volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR(14) for trend filter (strong move detection)
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
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Camarilla levels from previous day (using 1d data)
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_close_1d = np.roll(df_1d['close'].values, 1)
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    prev_close_1d[0] = df_1d['close'].iloc[0]
    prev_high_1d[0] = df_1d['high'].iloc[0]
    prev_low_1d[0] = df_1d['low'].iloc[0]
    
    camarilla_range = prev_high_1d - prev_low_1d
    r1 = prev_close_1d + camarilla_range * 1.1 / 12
    s1 = prev_close_1d - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20)  # need ATR14_1d, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > R1 (breakout resistance) AND ATR > 1.5x ATR MA (strong move) AND volume spike
            atr_ma = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean().values
            if (close[i] > r1_aligned[i] and 
                atr_14_1d_aligned[i] > 1.5 * atr_ma[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < S1 (breakdown support) AND ATR > 1.5x ATR MA (strong move) AND volume spike
            elif (close[i] < s1_aligned[i] and 
                  atr_14_1d_aligned[i] > 1.5 * atr_ma[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Close back inside previous day's Camarilla H-L range OR loss of momentum
            exit_signal = False
            if position == 1:
                # Exit long when close < S1 (breakdown of support) OR ATR < ATR MA (losing momentum)
                if close[i] < s1_aligned[i] or atr_14_1d_aligned[i] < atr_ma[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when close > R1 (breakout of resistance) OR ATR < ATR MA (losing momentum)
                if close[i] > r1_aligned[i] or atr_14_1d_aligned[i] < atr_ma[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R1S1_Breakout_1dATR_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0