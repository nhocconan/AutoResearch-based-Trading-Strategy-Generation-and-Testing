#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1wTrend_Volume
Hypothesis: Trading breakouts at inner Camarilla levels (R1/S1) from weekly pivot with 1-week EMA trend filter and volume confirmation. Weekly trend reduces false signals in sideways markets while capturing strong momentum moves. Target: 20-50 trades/year with low fee drag.
"""

name = "4h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter and pivot calculation
    df_1w = get_htf_ata(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate weekly Camarilla levels (R1/S1) using previous week's OHLC
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = high_1w[0]
    prev_low[0] = low_1w[0]
    prev_close[0] = close_1w[0]
    
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    
    # 4h data for signal generation
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA34 (34 periods)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs weekly EMA34
        uptrend_1w = close[i] > ema34_1w_aligned[i]
        downtrend_1w = close[i] < ema34_1w_aligned[i]
        
        # Volume filter: current volume > 1.5x 20-period EMA
        vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
        volume_filter = volume[i] > vol_ema20[i] * 1.5
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume and weekly uptrend
            if high[i] > R1_aligned[i] and uptrend_1w and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S1 with volume and weekly downtrend
            elif low[i] < S1_aligned[i] and downtrend_1w and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches midpoint between R1 and S1 or weekly trend fails
            midpoint = (R1_aligned[i] + S1_aligned[i]) / 2
            if low[i] <= midpoint or not uptrend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches midpoint between R1 and S1 or weekly trend fails
            midpoint = (R1_aligned[i] + S1_aligned[i]) / 2
            if high[i] >= midpoint or not downtrend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals