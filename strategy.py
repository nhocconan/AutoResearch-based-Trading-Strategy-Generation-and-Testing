#!/usr/bin/env python3
"""
12H_Camarilla_R1_S1_Breakout_1wTrend_Volume
Hypothesis: 12h breakouts at weekly (1w) Camarilla R1/S1 levels with volume confirmation and weekly EMA34 trend alignment capture directional moves in both bull and bear markets. Weekly trend filter ensures alignment with higher timeframe momentum, while volume confirmation filters false breakouts. Designed for low trade frequency (12-37/year) to minimize fee drag.
"""

name = "12H_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    
    # 1w data for Camarilla and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous 1w bar for Camarilla calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels from previous 1w bar
    range_1w = high_1w - low_1w
    s1 = close_1w - (range_1w * 1.08333)
    r1 = close_1w + (range_1w * 1.08333)
    
    # Align to 12h timeframe (wait for 1w bar to close)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    
    # Weekly trend filter: EMA 34
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: volume > 1.8x 20-period average (tight to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        is_uptrend = close[i] > ema_34_1w_aligned[i]
        is_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above R1 + volume confirmation + weekly uptrend
            if (close[i] > r1_aligned[i] and 
                volume[i] > vol_threshold[i] and 
                is_uptrend):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below S1 + volume confirmation + weekly downtrend
            elif (close[i] < s1_aligned[i] and 
                  volume[i] > vol_threshold[i] and 
                  is_downtrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses below S1 (opposite side)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above R1 (opposite side)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals