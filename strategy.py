#!/usr/bin/env python3
"""
12h_Williams_Alligator_Volume_Spike
Hypothesis: 12h Williams Alligator (jaw-teeth-lips crossover) with weekly trend filter and volume spike.
Williams Alligator uses smoothed medians to identify trends; works in both bull/bear by following the alignment.
Weekly trend filter avoids counter-trend trades. Volume spike confirms momentum.
Target: 12-30 trades/year on 12h to avoid fee drag.
"""

name = "12h_Williams_Alligator_Volume_Spike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get price, volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator (13,8,5 smoothed with 8,5,3)
    # Jaw (13-period SMMA smoothed by 8)
    sma13 = pd.Series(high + low).rolling(window=13, min_periods=13).mean().values / 2
    jaw = pd.Series(sma13).rolling(window=8, min_periods=8).mean().values
    
    # Teeth (8-period SMMA smoothed by 5)
    sma8 = pd.Series(high + low).rolling(window=8, min_periods=8).mean().values / 2
    teeth = pd.Series(sma8).rolling(window=5, min_periods=5).mean().values
    
    # Lips (5-period SMMA smoothed by 3)
    sma5 = pd.Series(high + low).rolling(window=5, min_periods=5).mean().values / 2
    lips = pd.Series(sma5).rolling(window=3, min_periods=3).mean().values
    
    # Volume filter: current volume > 2.0x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Alligator calculation (13+8=21) and weekly EMA (50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or
            np.isnan(lips[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND price above weekly EMA50 AND volume spike
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema_50_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Jaws > Teeth > Lips (bearish alignment) AND price below weekly EMA50 AND volume spike
            elif jaw[i] > teeth[i] and teeth[i] > lips[i] and close[i] < ema_50_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Teeth < Lips OR price below weekly EMA50
            if teeth[i] < lips[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Teeth > Lips OR price above weekly EMA50
            if teeth[i] > lips[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals