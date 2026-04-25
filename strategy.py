#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirm
Hypothesis: Trade 6h Donchian(20) breakouts only when aligned with weekly Camarilla pivot direction (S1/R1 for bias) and volume confirmation. Weekly pivot provides structural bias that works in both bull/bear markets by identifying key reversal/continuation levels from higher timeframe. Volume confirmation reduces false breakouts. Target: 12-25 trades/year per symbol.
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
    
    # Get 1w data for weekly Camarilla pivot bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (using previous weekly bar)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Camarilla H3/L3 and R1/S1 levels
    camarilla_h1w = close_1w + 1.1 * (high_1w - low_1w) / 4
    camarilla_l1w = close_1w - 1.1 * (high_1w - low_1w) / 4
    camarilla_r1w = close_1w + 1.1 * (high_1w - low_1w) / 2
    camarilla_s1w = close_1w - 1.1 * (high_1w - low_1w) / 2
    
    # Align weekly levels to 6h timeframe (completed weekly bar only)
    camarilla_h1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h1w)
    camarilla_l1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l1w)
    camarilla_r1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1w)
    camarilla_s1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1w)
    
    # 6h Donchian(20) channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20) and volume MA (20)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(camarilla_r1w_aligned[i]) or np.isnan(camarilla_s1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine weekly pivot bias: 
        # Bullish bias: price above weekly S1 (support holding)
        # Bearish bias: price below weekly R1 (resistance holding)
        weekly_bullish_bias = close[i] > camarilla_s1w_aligned[i]
        weekly_bearish_bias = close[i] < camarilla_r1w_aligned[i]
        
        if position == 0:
            # Long setup: Donchian breakout above weekly S1 bias + volume confirmation
            long_setup = (close[i] > high_ma[i]) and weekly_bullish_bias and volume_confirm[i]
            
            # Short setup: Donchian breakdown below weekly R1 bias + volume confirmation
            short_setup = (close[i] < low_ma[i]) and weekly_bearish_bias and volume_confirm[i]
            
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
            # Exit: price touches Donchian low OR weekly bias turns bearish
            if (close[i] <= low_ma[i]) or (not weekly_bullish_bias):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Donchian high OR weekly bias turns bullish
            if (close[i] >= high_ma[i]) or (weekly_bullish_bias):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirm"
timeframe = "6h"
leverage = 1.0