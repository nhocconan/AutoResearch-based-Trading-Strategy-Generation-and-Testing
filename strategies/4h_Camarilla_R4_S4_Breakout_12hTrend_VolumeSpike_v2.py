#!/usr/bin/env python3
"""
4h_Camarilla_R4_S4_Breakout_12hTrend_VolumeSpike_v2
Hypothesis: Building on previous success, tighten entry conditions to reduce trade frequency and improve robustness.
Uses Camarilla R4/S4 breakouts with volume confirmation (>2.5x average) and 12h EMA50 trend filter.
Reduces position size to 0.20 to lower drawdown risk. Target: 15-30 trades/year to avoid fee drag.
"""

name = "4h_Camarilla_R4_S4_Breakout_12hTrend_VolumeSpike_v2"
timeframe = "4h"
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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar (avoid look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla R4 and S4 levels
    rng = prev_high - prev_low
    r4 = prev_close + (rng * 1.1 / 2)  # R4 = C + (H-L) * 1.1/2
    s4 = prev_close - (rng * 1.1 / 2)  # S4 = C - (H-L) * 1.1/2
    
    # Align 1d levels to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation (20-period MA on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 12h EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher timeframe trend filter
        uptrend_12h = close[i] > ema_50_12h_aligned[i]
        downtrend_12h = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation (>2.5x average volume - stricter)
        volume_confirm = volume[i] > volume_ma[i] * 2.5
        
        if position == 0:
            # Long entry: uptrend + price breaks above R4 + volume confirmation
            if uptrend_12h and close[i] > r4_aligned[i] and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short entry: downtrend + price breaks below S4 + volume confirmation
            elif downtrend_12h and close[i] < s4_aligned[i] and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below R4
            if not uptrend_12h or close[i] < r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: trend breaks or price re-enters above S4
            if not downtrend_12h or close[i] > s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals