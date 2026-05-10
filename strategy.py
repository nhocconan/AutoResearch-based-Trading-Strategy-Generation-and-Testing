# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
4h_Pullback_to_MA_with_Volume_Confirmation
Hypothesis: In trending markets (defined by 1d EMA50), enter long/short on pullbacks to 4h EMA21 with volume confirmation.
The strategy works in both bull and bear regimes by following higher timeframe trend and entering on retracements.
Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
"""

name = "4h_Pullback_to_MA_with_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Get 4h price, volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA21 for pullback target
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50 bars), EMA21 (21), volume EMA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(ema21[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: above EMA50 (uptrend) AND price pulls back to EMA21 with volume
            if close[i] > ema_50_aligned[i] and low[i] <= ema21[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: below EMA50 (downtrend) AND price pulls back to EMA21 with volume
            elif close[i] < ema_50_aligned[i] and high[i] >= ema21[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below EMA21 OR trend turns bearish
            if close[i] < ema21[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above EMA21 OR trend turns bullish
            if close[i] > ema21[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals