#!/usr/bin/env python3
"""
1d_WilliamsAlligator_1wTrend_VolumeFilter
Hypothesis: Williams Alligator (jaw/teeth/lips) on 1d timeframe identifies trend alignment. 
When Alligator is "sleeping" (lines intertwined) we avoid trades; when "awakening" (lines diverging) 
we trade in direction of 1w trend with volume confirmation. Weekly trend filter avoids counter-trend 
trades in bear markets. Target: 7-25 trades/year per symbol.
Timeframe: 1d, HTF: 1w
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Calculate weekly EMA13 for trend filter (more responsive than 34)
    close_1w = df_1w['close'].values
    ema_13_1w = pd.Series(close_1w).ewm(span=13, min_periods=13, adjust=False).mean().values
    ema_13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    
    # Williams Alligator on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    close_1d = df_1d['close'].values
    
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw_raw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth_raw = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA, shifted 3 bars
    lips_raw = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan
    
    # Align Alligator lines to 1d timeframe (already 1d, but use for consistency)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume spike detector (20-period volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_13_1w_aligned[i]
        weekly_downtrend = close[i] < ema_13_1w_aligned[i]
        
        # Alligator sleeping condition: lines intertwined (max-min < 0.5% of price)
        alligator_range = np.max([jaw_aligned[i], teeth_aligned[i], lips_aligned[i]]) - \
                         np.min([jaw_aligned[i], teeth_aligned[i], lips_aligned[i]])
        alligator_sleeping = alligator_range < (close[i] * 0.005)
        
        # Alligator awakening: lips outside jaw/teeth
        lips_above = lips_aligned[i] > jaw_aligned[i] and lips_aligned[i] > teeth_aligned[i]
        lips_below = lips_aligned[i] < jaw_aligned[i] and lips_aligned[i] < teeth_aligned[i]
        
        if position == 0:
            # Long: Alligator awakening upward with volume spike and weekly uptrend
            if lips_above and volume_spike[i] and weekly_uptrend and not alligator_sleeping:
                signals[i] = 0.25
                position = 1
            # Short: Alligator awakening downward with volume spike and weekly downtrend
            elif lips_below and volume_spike[i] and weekly_downtrend and not alligator_sleeping:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Alligator sleeping again OR weekly trend changes to downtrend
            if alligator_sleeping or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Alligator sleeping again OR weekly trend changes to uptrend
            if alligator_sleeping or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WilliamsAlligator_1wTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0