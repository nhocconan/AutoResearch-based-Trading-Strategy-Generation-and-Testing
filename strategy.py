#!/usr/bin/env python3
# 4h_PivotPoint_Reversal_VolumeSpike_Trend
# Hypothesis: Price reversal at daily pivot points (PP, R1, S1) with volume spike
# and 12h EMA trend filter works in both bull and bear markets.
# Pivot points act as institutional support/resistance; volume confirms institutional interest.
# Trend filter prevents counter-trend trades in strong moves.
# Target: 20-40 trades/year on 4h timeframe.

name = "4h_PivotPoint_Reversal_VolumeSpike_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivot_points(high, low, close):
    """Calculate daily pivot points and support/resistance levels"""
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (high + low + close) / 3
    # Resistance 1 (R1) = (2 * PP) - Low
    r1 = (2 * pp) - low
    # Support 1 (S1) = (2 * PP) - High
    s1 = (2 * pp) - high
    return pp, r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily pivot points
    pp_1d, r1_1d, s1_1d = calculate_pivot_points(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align pivot points to 4h timeframe
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2.0 x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need pivot points (30) + EMA50 (50) + volume EMA (20)
    start_idx = max(30, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pp_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine if we are in uptrend or downtrend based on 12h EMA50
        is_uptrend = close[i] > ema_50_12h_aligned[i]
        is_downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long setup: price at or below S1 with volume spike in uptrend
            if is_uptrend and low[i] <= s1_1d_aligned[i] * 1.001 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short setup: price at or above R1 with volume spike in downtrend
            elif is_downtrend and high[i] >= r1_1d_aligned[i] * 0.999 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses above PP or volume dries up
            if close[i] > pp_1d_aligned[i] or volume[i] < vol_ema20[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses below PP or volume dries up
            if close[i] < pp_1d_aligned[i] or volume[i] < vol_ema20[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals