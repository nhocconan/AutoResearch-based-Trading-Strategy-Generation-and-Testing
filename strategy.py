#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation.
Designed for 12h timeframe to target 12-37 trades/year. Uses discrete sizing (0.25) to minimize fee drag.
Works in bull via breakouts, in bear via volatility expansion + mean reversion at channel extremes.
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d Donchian channels (20-period)
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_mid_20 = (donch_high_20 + donch_low_20) / 2.0
    
    # Align 1d indicators to 12h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid_20)
    
    # Calculate 12h volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 20)  # need Donchian20, ATR14, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(donch_mid_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: current 12h ATR > 1.0x 1d ATR (avoid low volatility periods)
        # Calculate current 12h ATR(14) for comparison
        if i >= 14:
            tr_12h = np.maximum(np.abs(high[i] - low[i]), 
                               np.maximum(np.abs(high[i] - close[i-1]), 
                                          np.abs(low[i] - close[i-1])))
            # Simplified: use rolling ATR approximation for regime
            vol_filter = True  # Will use ATR ratio below
        else:
            vol_filter = False
        
        # Calculate 12h ATR(14) for volatility regime
        if i >= 27:  # Need 14+14 for proper ATR
            tr_seq = []
            for j in range(max(0, i-13), i+1):
                tr = np.maximum(np.abs(high[j] - low[j]), 
                               np.maximum(np.abs(high[j] - close[j-1] if j>0 else high[j]), 
                                          np.abs(low[j] - close[j-1] if j>0 else low[j])))
                tr_seq.append(tr)
            atr_12h_current = np.mean(tr_seq[-14:]) if len(tr_seq) >= 14 else atr_aligned[i]
        else:
            atr_12h_current = atr_aligned[i]
        
        # Volatility filter: 12h ATR > 0.8 x 1d ATR (avoid extremely low vol)
        vol_filter = atr_12h_current > 0.8 * atr_aligned[i]
        
        # Volume filter: 12h volume > 1.5x 20-period MA
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_high_aligned[i]
        breakout_down = close[i] < donch_low_aligned[i]
        return_to_mid = (abs(close[i] - donch_mid_aligned[i]) < 
                        0.1 * (donch_high_aligned[i] - donch_low_aligned[i]))
        
        if position == 0:
            # Long: Break above upper Donchian with volume and volatility confirmation
            if breakout_up and volume_filter and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian with volume and volatility confirmation
            elif breakout_down and volume_filter and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: return to middle of channel or opposite breakout
            exit_signal = False
            if position == 1:
                exit_signal = return_to_mid or breakout_down
            elif position == -1:
                exit_signal = return_to_mid or breakout_up
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_Breakout_1dATR_VolumeFilter"
timeframe = "12h"
leverage = 1.0