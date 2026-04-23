#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian channel breakout with 1-day ATR filter and volume confirmation.
Long when price breaks above Donchian(20) high, ATR(14) > 1.2 * ATR(50), and volume > 1.3 * average volume.
Short when price breaks below Donchian(20) low, ATR(14) > 1.2 * ATR(50), and volume > 1.3 * average volume.
Exit when price returns to Donchian midpoint or ATR condition fails.
Designed for low trade frequency (~20-40/year) to capture strong volatility breakouts while avoiding chop.
Works in both bull and bear markets by requiring volatility expansion (ATR ratio > 1.2) to confirm genuine breakouts.
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
    
    # Load 1-day data for ATR - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1-day ATR(14) and ATR(50)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR(14)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # ATR(50)
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # ATR ratio: short-term / long-term
    atr_ratio = np.where(atr50 > 0, atr14 / atr50, 0.0)
    
    # Align ATR ratio to lower timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Donchian channel (20-period) on 4h
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Volume average (20-period) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr_ratio_val = atr_ratio_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Donchian breakout high, volatility expansion, volume confirmation
            if (close[i] > donch_high[i-1] and 
                atr_ratio_val > 1.2 and 
                vol_current > 1.3 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout low, volatility expansion, volume confirmation
            elif (close[i] < donch_low[i-1] and 
                  atr_ratio_val > 1.2 and 
                  vol_current > 1.3 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to midpoint OR volatility contraction
                if (close[i] <= donch_mid[i] or atr_ratio_val < 1.0):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to midpoint OR volatility contraction
                if (close[i] >= donch_mid[i] or atr_ratio_val < 1.0):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_ATR_Volume"
timeframe = "4h"
leverage = 1.0