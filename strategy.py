# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeConfirm_ChopFilter_v1
Hypothesis: Donchian(20) breakout with volume confirmation and chop filter.
Breakouts are significant only when accompanied by volume and non-choppy markets.
This reduces false breakouts and works in both trending (breakout) and ranging (avoid) regimes.
Target: 20-50 trades/year for low fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Chop filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((highest_high - lowest_low) / np.sum(tr[-14:])) if len(tr) >= 14 else 50
    # Fix chop calculation: use rolling sum
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10((highest_high - lowest_low) / tr_sum)
    chop = np.where(tr_sum > 0, chop, 50)  # avoid div/0
    
    # Volume confirmation: current > 1.5 * 20-period avg
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(chop[i]) or np.isnan(volume_confirm[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper band with volume and low chop (trending)
            if close[i] > high_max[i] and volume_confirm[i] and chop[i] < 61.8:
                signals[i] = size
                position = 1
            # Short: break below lower band with volume and low chop
            elif close[i] < low_min[i] and volume_confirm[i] and chop[i] < 61.8:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit: close below midpoint or chop increases (range)
            midpoint = (high_max[i] + low_min[i]) / 2.0
            if close[i] < midpoint or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: close above midpoint or chop increases
            midpoint = (high_max[i] + low_min[i]) / 2.0
            if close[i] > midpoint or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_VolumeConfirm_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0