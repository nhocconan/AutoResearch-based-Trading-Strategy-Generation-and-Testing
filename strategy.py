#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_VolumeTrend
Hypothesis: Using 12h timeframe with Camarilla pivot levels from 1d (R4/S4) for breakout entries, 
confirmed by volume spikes (>2x average) and filtered by 1d EMA34 trend. 
Designed for fewer trades (target 12-37/year) to minimize fee drag and work in both bull/bear markets via trend filter.
"""

name = "12h_Camarilla_Pivot_VolumeTrend"
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
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar (avoid look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla R4 and S4 levels
    rng = prev_high - prev_low
    r4 = prev_close + (rng * 1.1 / 2)  # R4 = C + (H-L) * 1.1/2
    s4 = prev_close - (rng * 1.1 / 2)  # S4 = C - (H-L) * 1.1/2
    
    # Align 1d levels to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period MA on 12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d EMA34 (34) and volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher timeframe trend filter
        uptrend_1d = close[i] > ema_34_1d_aligned[i]
        downtrend_1d = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation (>2x average volume)
        volume_confirm = volume[i] > volume_ma[i] * 2.0
        
        if position == 0:
            # Long entry: uptrend + price breaks above R4 + volume confirmation
            if uptrend_1d and close[i] > r4_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price breaks below S4 + volume confirmation
            elif downtrend_1d and close[i] < s4_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below R4
            if not uptrend_1d or close[i] < r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters above S4
            if not downtrend_1d or close[i] > s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals