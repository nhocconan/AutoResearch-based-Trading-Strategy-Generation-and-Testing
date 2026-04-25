#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dATR_Trend_VolumeSp
Hypothesis: Trade Donchian(20) breakouts on 4h with 1d ATR-based trend filter and volume confirmation.
Uses 1d ATR to define trend strength (ATR rising = trending market) and reduce false breakouts in choppy/low-vol regimes.
Volume confirmation ensures breakouts have conviction. Discrete sizing (0.25) limits fee drag.
Designed to work in both bull and bear markets by filtering for trending conditions via rising ATR.
Target: 20-50 trades/year per symbol.
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
    
    # Get 1d data for HTF trend filter (ATR-based)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Trend filter: ATR rising (current > previous) indicates strengthening trend
    atr_rising = atr_14_1d > np.roll(atr_14_1d, 1)
    atr_rising[0] = False  # first bar undefined
    atr_rising_aligned = align_htf_to_ltf(prices, df_1d, atr_rising)
    
    # Calculate Donchian(20) on 4h
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian(20), volume MA(20), and ATR(14) -> need 20 bars
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_rising_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long setup: price breaks above Donchian H + ATR rising (trending up) + volume confirmation
            long_setup = (close[i] > donchian_h[i]) and atr_rising_aligned[i] and volume_confirm[i]
            
            # Short setup: price breaks below Donchian L + ATR rising (trending down) + volume confirmation
            short_setup = (close[i] < donchian_l[i]) and atr_rising_aligned[i] and volume_confirm[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches Donchian L (stop) OR ATR stops rising (trend weakening)
            if (close[i] <= donchian_l[i]) or (not atr_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Donchian H (stop) OR ATR stops rising (trend weakening)
            if (close[i] >= donchian_h[i]) or (not atr_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dATR_Trend_VolumeSp"
timeframe = "4h"
leverage = 1.0