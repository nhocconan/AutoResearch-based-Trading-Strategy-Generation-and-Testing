#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian breakout with 1-day ATR filter and volume confirmation.
Long when price breaks above 20-period high with expanding volatility and volume spike.
Short when price breaks below 20-period low with expanding volatility and volume spike.
Exit when price returns to midline or volatility contracts.
Designed for low trade frequency by requiring volatility expansion and volume confirmation.
Works in both bull and bear markets by capturing breakouts in trending phases.
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
    
    # Load 12h data for Donchian channels - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 20-period Donchian channels on 12h
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Align to 1h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_12h, donch_mid)
    
    # Load 1d data for ATR filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 14-period ATR on 1d
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align ATR to 1h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Volume confirmation: current volume > 1.5x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(vol_ma_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility expansion: current ATR > 1.2x ATR from 5 periods ago
        vol_expanding = atr_aligned[i] > 1.2 * atr_aligned[i-5] if i >= 5 else False
        
        # Volume spike
        vol_spike = volume[i] > 1.5 * vol_ma_30[i]
        
        if position == 0:
            # Long: break above Donchian high with vol expansion and volume spike
            if close[i] > donch_high_aligned[i] and vol_expanding and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with vol expansion and volume spike
            elif close[i] < donch_low_aligned[i] and vol_expanding and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: return to midline or volatility contraction
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to midline or volatility contracts
                if close[i] <= donch_mid_aligned[i] or (i >= 5 and atr_aligned[i] < 0.8 * atr_aligned[i-5]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to midline or volatility contracts
                if close[i] >= donch_mid_aligned[i] or (i >= 5 and atr_aligned[i] < 0.8 * atr_aligned[i-5]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_DonchianBreakout_1dATR_Volume"
timeframe = "12h"
leverage = 1.0