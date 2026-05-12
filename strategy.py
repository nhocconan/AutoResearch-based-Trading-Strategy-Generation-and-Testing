#!/usr/bin/env python3
"""
6H_EMA200_RETEST_WITH_VOLUME
Hypothesis: Price retests the 200-period EMA on 6h chart during strong trends, offering high-probability entries.
Combines EMA200 trend filter (from 1d timeframe for multi-timeframe alignment) with volume confirmation to avoid false breakouts.
Works in both bull and bear markets by trading in direction of higher timeframe trend.
Target: 20-40 trades per year to minimize fee drag.
"""

name = "6H_EMA200_RETEST_WITH_VOLUME"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA200 on 6h for trend definition (but we'll use 1d EMA200 for higher timeframe alignment)
    ema200_6h = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 1d EMA200 for multi-timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    pclose_1d = df_1d['close'].values
    ema200_1d = pd.Series(pclose_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200 to be stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema200_6h[i]) or np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            # LONG: Price touches or crosses above EMA200 from below with volume and uptrend on 1d
            if close[i] >= ema200_6h[i] and close[i-1] < ema200_6h[i-1] and vol_confirm and ema200_1d_aligned[i] < ema200_1d_aligned[i-1]:
                # Additional confirmation: EMA200 rising on 1d (uptrend)
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches or crosses below EMA200 from above with volume and downtrend on 1d
            elif close[i] <= ema200_6h[i] and close[i-1] > ema200_6h[i-1] and vol_confirm and ema200_1d_aligned[i] > ema200_1d_aligned[i-1]:
                # Additional confirmation: EMA200 falling on 1d (downtrend)
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below EMA200 or volume drops significantly
            if close[i] < ema200_6h[i] or volume[i] < vol_ma[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above EMA200 or volume drops significantly
            if close[i] > ema200_6h[i] or volume[i] < vol_ma[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals