#!/usr/bin/env python3
"""
6H_WeeklyPivot_Power_Trend_1d_Volume
Hypothesis: Combines weekly pivot levels (PP, R1, S1) with 1d Elder Ray power (bull/bear power) and volume confirmation. 
Enters long when price > weekly PP + bull power > 0 + volume spike, short when price < weekly PP + bear power < 0 + volume spike.
Uses weekly trend filter (price > weekly EMA20 for long, < for short) to avoid counter-trend trades.
Designed for 6h timeframe to capture multi-day momentum with controlled trade frequency.
Works in bull/bear by following weekly trend and using mean-reversion at weekly pivots in ranging markets.
"""

name = "6H_WeeklyPivot_Power_Trend_1d_Volume"
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
    
    # 1d data for Elder Ray power calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly data for pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1d Elder Ray power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    # Calculate weekly pivot points from previous weekly bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Pivot Point (PP) = (High + Low + Close) / 3
    pp = (high_1w + low_1w + close_1w) / 3.0
    # R1 = 2*PP - Low
    r1 = 2 * pp - low_1w
    # S1 = 2*PP - High
    s1 = 2 * pp - high_1w
    
    # Align weekly pivot points to 6h timeframe (wait for weekly bar to close)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Align 1d Elder Ray power to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Weekly trend filter: EMA 20
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume filter: volume > 2.0x 20-period average (strict to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        is_uptrend = close[i] > ema_20_1w_aligned[i]
        is_downtrend = close[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long entry: Price above weekly PP + bull power positive + volume spike + weekly uptrend
            if (close[i] > pp_aligned[i] and 
                bull_power_aligned[i] > 0 and 
                volume[i] > vol_threshold[i] and 
                is_uptrend):
                signals[i] = 0.25
                position = 1
            # Short entry: Price below weekly PP + bear power negative + volume spike + weekly downtrend
            elif (close[i] < pp_aligned[i] and 
                  bear_power_aligned[i] < 0 and 
                  volume[i] > vol_threshold[i] and 
                  is_downtrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses below weekly S1 OR bull power turns negative
            if close[i] < s1_aligned[i] or bull_power_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above weekly R1 OR bear power turns positive
            if close[i] > r1_aligned[i] or bear_power_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals