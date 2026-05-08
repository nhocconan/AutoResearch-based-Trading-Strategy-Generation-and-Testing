# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d pivot-based mean reversion in ranging markets.
Uses Camarilla pivot levels (H3/L3) as mean reversion zones, filtered by 1d trend
and volume spikes to avoid false signals in strong trends. Designed for 50-150
trades over 4 years (12-37/year) with discrete position sizing to minimize fee drag.
Works in both bull/bear by adapting to 1d trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_H3L3_MeanReversion_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels: H3/L3 (mean reversion zones)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    h3 = close_1d + (range_1d * 1.1 / 4)
    l3 = close_1d - (range_1d * 1.1 / 4)
    
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price near L3 (support) + uptrend + volume spike
            long_cond = (close[i] <= l3_aligned[i] * 1.005) and \
                        (close[i] > ema_34_1d_aligned[i]) and \
                        volume_spike[i]
            # Short: price near H3 (resistance) + downtrend + volume spike
            short_cond = (close[i] >= h3_aligned[i] * 0.995) and \
                         (close[i] < ema_34_1d_aligned[i]) and \
                         volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price moves to midpoint or H3 (take profit)
            if close[i] >= (l3_aligned[i] + h3_aligned[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price moves to midpoint or L3 (take profit)
            if close[i] <= (l3_aligned[i] + h3_aligned[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals