#!/usr/bin/env python3
# 4h_1d_Camarilla_R1S1_Breakout_VolumeTrend
# Hypothesis: 4h breakouts at tight Camarilla R1/S1 levels from 1d, filtered by 1d trend (EMA50) and volume surge.
# Uses R1/S1 (narrower than R3/S3) for higher precision entries. 1d trend filter ensures alignment with daily momentum.
# Volume surge confirms breakout strength. Designed for low trade frequency (<50/year) to minimize fee drag.
# Works in bull/bear markets by requiring trend alignment and volume confirmation.

name = "4h_1d_Camarilla_R1S1_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla R1/S1 levels from previous 1d bar (H, L, C)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1/S1 levels (tighter than R3/S3)
    R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align 1d Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Camarilla (2) + volume MA (20) + EMA (50)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA50
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        # Price position relative to Camarilla levels
        price_above_R1 = close[i] > R1_aligned[i]
        price_below_S1 = close[i] < S1_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume surge and 1d uptrend
            if price_above_R1 and volume_surge and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume surge and 1d downtrend
            elif price_below_S1 and volume_surge and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls back below R1 OR trend changes
            if close[i] < R1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises back above S1 OR trend changes
            if close[i] > S1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals