#!/usr/bin/env python3
"""
12h_Williams_Alligator_Trend_1dTrend_Filter
Hypothesis: Use Williams Alligator (SMAs) on 12h for trend direction, filtered by 1d EMA trend and volume confirmation. 
Alligator provides clear trend signals (jaw-teeth-lips alignment) that work in both bull/bear markets when aligned with higher timeframe trend.
Target: 15-25 trades/year on 12h to stay within fee limits.
"""

name = "12h_Williams_Alligator_Trend_1dTrend_Filter"
timeframe = "12h"
leverage = 1.0

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
    
    # === 1d Higher Timeframe Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Williams Alligator on 12h (using SMAs) ===
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # Using SMA as approximation for SMMA (Smoothed Moving Average)
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator components (already on 12h timeframe)
    jaw = jaw_raw
    teeth = teeth_raw
    lips = lips_raw
    
    # Volume filter: 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema34_12h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + uptrend + volume
            if (lips[i] > teeth[i] > jaw[i] and 
                close[i] > ema34_12h[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short: Jaws > Teeth > Lips (bearish alignment) + downtrend + volume
            elif (jaw[i] > teeth[i] > lips[i] and 
                  close[i] < ema34_12h[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator starts to converge or trend changes
            if (lips[i] <= teeth[i] or  # Lips cross below teeth
                close[i] < ema34_12h[i]):  # Price breaks below daily trend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: Alligator starts to converge or trend changes
            if (teeth[i] <= lips[i] or  # Teeth cross below lips
                close[i] > ema34_12h[i]):  # Price breaks above daily trend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals