#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeSpike
Hypothesis: On 1h timeframe, trade breakouts of daily Camarilla R3/S3 levels with 4h EMA50 trend filter and volume spike confirmation.
Uses 4h for signal direction, 1h for entry timing. Session filter (08-20 UTC) reduces noise. Target: 15-37 trades/year per symbol.
"""

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla formulas
    R3 = close_prev + (high_prev - low_prev) * 1.1 / 4
    S3 = close_prev - (high_prev - low_prev) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (need previous day's levels)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Get 1h data for volume and price
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: current volume > 2x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > vol_ema20 * 2.0
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # Track bars since last entry to enforce min hold
    
    # Warmup: need 4h EMA (50) and Camarilla (need 2 days for shift)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                bars_since_entry = 0
            continue
        
        # Skip if outside session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                bars_since_entry = 0
            continue
        
        # Increment bars since entry if in a position
        if position != 0:
            bars_since_entry += 1
        
        # Trend filter: price vs 4h EMA50
        uptrend_4h = close[i] > ema50_4h_aligned[i]
        downtrend_4h = close[i] < ema50_4h_aligned[i]
        
        if position == 0:
            # Long: break above R3 in uptrend with volume spike
            if high[i] > R3_aligned[i] and uptrend_4h and volume_spike[i]:
                signals[i] = 0.20
                position = 1
                bars_since_entry = 0
            # Short: break below S3 in downtrend with volume spike
            elif low[i] < S3_aligned[i] and downtrend_4h and volume_spike[i]:
                signals[i] = -0.20
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Long exit: price drops below S3 or trend fails
            if low[i] < S3_aligned[i] or not uptrend_4h:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price rises above R3 or trend fails
            if high[i] > R3_aligned[i] or not downtrend_4h:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.20
    
    return signals