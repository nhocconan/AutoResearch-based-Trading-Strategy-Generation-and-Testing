#!/usr/bin/env python3
"""
6h_PivotReversal_1wTrend_VolumeSpike
Hypothesis: On 6h timeframe, trade reversals from 1d Camarilla R3/S3 levels when aligned with 1w trend (EMA50) and confirmed by volume spike. 
Only long at S3 in 1w uptrend, short at R3 in 1w downtrend. Uses discrete sizing (0.25) to minimize fee drag. 
Target: 12-30 trades/year per symbol. Works in bull/bear via 1w trend filter - avoids counter-trend trades.
"""

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
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla width
    rang = high_1d - low_1d
    
    # Resistance levels
    r3 = close_1d_prev + rang * 1.1 / 4
    r4 = close_1d_prev + rang * 1.1 / 2
    
    # Support levels
    s3 = close_1d_prev - rang * 1.1 / 4
    s4 = close_1d_prev - rang * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume spike detector (20-bar volume MA on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter: 1w EMA50 direction
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks below S3 (mean reversion) in 1w uptrend with volume spike
            if close[i] < s3_aligned[i] and uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks above R3 (mean reversion) in 1w downtrend with volume spike
            elif close[i] > r3_aligned[i] and downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes above S3 (reversion played out) OR trend changes
            if close[i] > s3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes below R3 (reversion played out) OR trend changes
            if close[i] < r3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_PivotReversal_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0