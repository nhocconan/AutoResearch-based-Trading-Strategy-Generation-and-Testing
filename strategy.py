#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels from 1d (R3/S3) act as strong support/resistance.
Breakout above R3 with 1d uptrend (close > EMA34) and volume spike triggers long.
Breakdown below S3 with 1d downtrend (close < EMA34) and volume spike triggers short.
Designed for low-frequency, high-conviction trades (target 50-150 over 4 years) on 12h.
Works in bull (breakouts continue) and bear (breakdowns continue) via trend filter.
"""

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # === 1D Data for Camarilla Pivots and Trend ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    R3 = np.zeros_like(close_1d)
    S3 = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            # First day: use same day's range
            R3[i] = close_1d[i] + (high_1d[i] - low_1d[i]) * 1.1 / 4
            S3[i] = close_1d[i] - (high_1d[i] - low_1d[i]) * 1.1 / 4
        else:
            # Use previous day's range (standard Camarilla)
            R3[i] = close_1d[i-1] + (high_1d[i-1] - low_1d[i-1]) * 1.1 / 4
            S3[i] = close_1d[i-1] - (high_1d[i-1] - low_1d[i-1]) * 1.1 / 4
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d data to 12h
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Spike Detector (20-period average) ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 35  # need EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R3 with uptrend and volume spike
            if (close[i] > R3_aligned[i] and 
                close_1d[-1] > ema34_1d_aligned[i] if len(close_1d) > 0 else False and  # current 1d close > EMA34
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with downtrend and volume spike
            elif (close[i] < S3_aligned[i] and 
                  close_1d[-1] < ema34_1d_aligned[i] if len(close_1d) > 0 else False and  # current 1d close < EMA34
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below S3 (reversal signal) OR trend flip
            if close[i] < S3_aligned[i] or close_1d[-1] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: close above R3 (reversal signal) OR trend flip
            if close[i] > R3_aligned[i] or close_1d[-1] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals