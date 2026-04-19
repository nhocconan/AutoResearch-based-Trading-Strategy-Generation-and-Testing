# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + 1d Trend Filter + Volume Confirmation
- Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength
- 1d EMA34 trend filter ensures alignment with higher timeframe trend
- Volume spike (>2x 20-period average) confirms conviction
- Designed to capture strong trends while avoiding choppy markets
- Target: 15-35 trades/year to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsAlligator_1dTrendFilter_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 6h
    # Jaw (13-period SMMA, 8 periods ahead)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (8-period SMMA, 5 periods ahead)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (5-period SMMA, 3 periods ahead)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x average
        volume_filter = volume[i] > 2 * vol_ma[i]
        
        # Alligator alignment check
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        alligator_long = lips_above_teeth and teeth_above_jaw
        alligator_short = lips_below_teeth and teeth_below_jaw
        
        if position == 0:
            # Long entry: Alligator aligned up + 1d uptrend + volume
            if alligator_long and close[i] > ema_34_1d_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: Alligator aligned down + 1d downtrend + volume
            elif alligator_short and close[i] < ema_34_1d_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator turns down OR price crosses below 1d EMA
            if not alligator_long or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator turns up OR price crosses above 1d EMA
            if not alligator_short or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals