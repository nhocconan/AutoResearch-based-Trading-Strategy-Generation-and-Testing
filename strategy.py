#!/usr/bin/env python3
"""
12h_Pivot_R1_S1_1dTrend_Volume
Hypothesis: On 12h timeframe, price touching Camarilla R1 or S1 levels derived from prior 1d candle,
combined with 1d trend filter (EMA34) and volume confirmation, provides high-probability mean-reversion
entries in range-bound markets and breakout entries in trending markets. Works in bull/bear via
adaptive interpretation of pivot levels: in uptrend, S1 acts as support for longs; in downtrend,
R1 acts as resistance for shorts. Designed for low trade frequency (target: 12-37/year) to minimize fee drag.
"""

name = "12h_Pivot_R1_S1_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12.0
    r1_1d = close_1d + camarilla_range
    s1_1d = close_1d - camarilla_range
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d volume moving average
    vol_ma_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all 1d indicators to 12h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 12h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 (34)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1d trend filter
        uptrend_1d = close[i] > ema34_1d_aligned[i]
        downtrend_1d = close[i] < ema34_1d_aligned[i]
        
        # Volume filter: current volume > 1.5x 1d average volume (scaled to 12h)
        # 1d = 2 x 12h bars, so scale 1d volume to 12h equivalent
        vol_12h_equiv = vol_ma_1d_aligned[i] / 2.0
        volume_filter = volume[i] > vol_12h_equiv * 1.5
        
        if position == 0:
            # Long entry: price touches or goes below S1 + uptrend + volume
            # In uptrend, S1 acts as support for long entries
            if low[i] <= s1_1d_aligned[i] and uptrend_1d and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price touches or goes above R1 + downtrend + volume
            # In downtrend, R1 acts as resistance for short entries
            elif high[i] >= r1_1d_aligned[i] and downtrend_1d and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches or crosses R1 or trend fails
            if high[i] >= r1_1d_aligned[i] or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches or crosses S1 or trend fails
            if low[i] <= s1_1d_aligned[i] or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals