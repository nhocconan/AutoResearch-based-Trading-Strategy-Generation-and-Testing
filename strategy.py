#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike
Hypothesis: On 6h timeframe, use Camarilla pivot levels from 1d to identify breakout opportunities at R3/S3 levels, filtered by 12h EMA50 trend direction and volume spike confirmation. This strategy aims to capture momentum breakouts with proper trend alignment and volume confirmation, working in both bull and bear markets by following the higher timeframe trend. The Camarilla levels provide precise entry points, while the 12h trend filter ensures we trade with the dominant trend, and volume spike confirms institutional interest.
"""

name = "6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3, R4, S3, S4
    # R3 = Close + (High - Low) * 1.1/2
    # R4 = Close + (High - Low) * 1.1
    # S3 = Close - (High - Low) * 1.1/2
    # S4 = Close - (High - Low) * 1.1
    rng = high_1d - low_1d
    r3 = close_1d + rng * 1.1 / 2
    r4 = close_1d + rng * 1.1
    s3 = close_1d - rng * 1.1 / 2
    s4 = close_1d - rng * 1.1
    
    # Align Camarilla levels to 6h timeframe (using previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Get 1d data for volume filter
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # 6h data for price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need at least 1 day of data for pivots, 50 for EMA, 20 for volume MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or
            np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs 12h EMA50
        uptrend_12h = close[i] > ema50_12h_aligned[i]
        downtrend_12h = close[i] < ema50_12h_aligned[i]
        
        # Volume filter: current 6h volume > 2x 1d 20-period MA
        volume_filter = volume[i] > vol_ma20_1d_aligned[i] * 2.0
        
        if position == 0:
            # Long breakout: price breaks above R3 with volume and uptrend
            if close[i] > r3_aligned[i] and volume_filter and uptrend_12h:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S3 with volume and downtrend
            elif close[i] < s3_aligned[i] and volume_filter and downtrend_12h:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below R4 (failed breakout) or trend fails
            if close[i] < r4_aligned[i] or not uptrend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above S4 (failed breakdown) or trend fails
            if close[i] > s4_aligned[i] or not downtrend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals