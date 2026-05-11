#!/usr/bin/env python3
name = "6h_1d_ElderRay_ForceIndex_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d Elder Ray and Force Index components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 13-period EMA for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Force Index = (Close - Previous Close) * Volume
    # Need previous close, handle first element
    price_change_1d = np.diff(close_1d, prepend=close_1d[0])
    force_index_1d = price_change_1d * volume_1d
    
    # 2-period EMA of Force Index for smoothing
    force_ema2_1d = pd.Series(force_index_1d).ewm(span=2, adjust=False, min_periods=2).mean().values
    
    # Align all 1d indicators to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    force_ema2_1d_aligned = align_htf_to_ltf(prices, df_1d, force_ema2_1d)
    
    # 6m EMA for trend filter (13-period)
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 13
    
    for i in range(start_idx, n):
        if np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or np.isnan(force_ema2_1d_aligned[i]) or np.isnan(ema13_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND Force Index rising AND price above EMA13
            if bull_power_1d_aligned[i] > 0 and force_ema2_1d_aligned[i] > force_ema2_1d_aligned[i-1] and close[i] > ema13_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Force Index falling AND price below EMA13
            elif bear_power_1d_aligned[i] < 0 and force_ema2_1d_aligned[i] < force_ema2_1d_aligned[i-1] and close[i] < ema13_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power < 0 OR Force Index falling
            if bear_power_1d_aligned[i] < 0 or force_ema2_1d_aligned[i] < force_ema2_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power > 0 OR Force Index rising
            if bull_power_1d_aligned[i] > 0 or force_ema2_1d_aligned[i] > force_ema2_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals