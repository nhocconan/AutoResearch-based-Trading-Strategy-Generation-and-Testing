#!/usr/bin/env python3
"""
4h_ThreeBarReversal_1dTrend_VolumeConfirm
Hypothesis: Trade three-bar reversal patterns (bullish/bearish) aligned with daily trend and volume confirmation. Works in both bull/bear by filtering with daily EMA trend. Target: 20-40 trades/year on 4h.
"""

name = "4h_ThreeBarReversal_1dTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily Trend Filter (EMA34) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Filter (1.5x 20-period EMA on 4h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema34_4h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Bullish 3-bar reversal: low > previous low for 3 consecutive bars
            bullish_reversal = (low[i] > low[i-1] and 
                               low[i-1] > low[i-2] and 
                               low[i-2] > low[i-3])
            
            # Bearish 3-bar reversal: high < previous high for 3 consecutive bars
            bearish_reversal = (high[i] < high[i-1] and 
                               high[i-1] < high[i-2] and 
                               high[i-2] < high[i-3])
            
            # Long: bullish reversal with uptrend and volume
            if bullish_reversal and close[i] > ema34_4h[i] and volume_ok[i]:
                signals[i] = 0.30
                position = 1
            # Short: bearish reversal with downtrend and volume
            elif bearish_reversal and close[i] < ema34_4h[i] and volume_ok[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: bearish reversal or price below EMA
            bearish_reversal = (high[i] < high[i-1] and 
                               high[i-1] < high[i-2] and 
                               high[i-2] < high[i-3])
            if bearish_reversal or close[i] < ema34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30  # maintain position
        elif position == -1:
            # Short exit: bullish reversal or price above EMA
            bullish_reversal = (low[i] > low[i-1] and 
                               low[i-1] > low[i-2] and 
                               low[i-2] > low[i-3])
            if bullish_reversal or close[i] > ema34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30  # maintain position
    
    return signals