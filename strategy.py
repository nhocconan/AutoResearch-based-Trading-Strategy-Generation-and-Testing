#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: Using weekly trend filter (1w EMA200) with daily Camarilla R1/S1 breakouts on 12h timeframe.
Weekly trend filter provides strong directional bias, reducing false breakouts. Daily Camarilla levels provide
intraday support/resistance. Volume confirmation (>2x average) ensures breakout strength. Target: 20-40 trades/year.
"""

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous daily bar (R1/S1)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R1 and S1 levels
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.1 / 12)  # R1 = C + (H-L) * 1.1/12
    s1 = prev_close - (rng * 1.1 / 12)  # S1 = C - (H-L) * 1.1/12
    
    # Align daily levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate weekly EMA200 for trend filter
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume confirmation (30-period MA on 12h)
    volume_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA200 (200) and volume MA (30)
    start_idx = max(200, 30)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend_1w = close[i] > ema_200_1w_aligned[i]
        downtrend_1w = close[i] < ema_200_1w_aligned[i]
        
        # Volume confirmation (>2x average volume)
        volume_confirm = volume[i] > volume_ma[i] * 2.0
        
        if position == 0:
            # Long entry: weekly uptrend + price breaks above R1 + volume confirmation
            if uptrend_1w and close[i] > r1_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: weekly downtrend + price breaks below S1 + volume confirmation
            elif downtrend_1w and close[i] < s1_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly trend breaks or price re-enters below R1
            if not uptrend_1w or close[i] < r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly trend breaks or price re-enters above S1
            if not downtrend_1w or close[i] > s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals