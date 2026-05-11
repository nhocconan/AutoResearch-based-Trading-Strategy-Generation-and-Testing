#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Fade_Backtest_v1
Hypothesis: Fades price at Camarilla R3/S3 levels (strong intraday support/resistance)
with confirmation from 1d trend (EMA34). In ranging markets, price reverses at these levels;
in trending markets, we avoid false reversals by requiring counter-trend alignment.
Uses volume spike to confirm institutional interest at the level.
Target: 50-150 trades over 4 years (12-37/year) on 6h timeframe.
"""

name = "6h_Camarilla_R3_S3_Fade_Backtest_v1"
timeframe = "6h"
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
    
    # === 1D Data for Camarilla Calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    rangec = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * rangec / 2
    camarilla_s3 = close_1d - 1.1 * rangec / 2
    
    # Align Camarilla levels to 6h
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === 1D Trend Filter (EMA34) ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Spike Detection (20-period average) ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(100, 34)
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long setup: price at or below S3 with volume spike in uptrend (mean reversion long)
            if close[i] <= s3_aligned[i] and vol_spike[i] and ema34_1d_aligned[i] < close[i]:
                signals[i] = 0.25
                position = 1
            # Short setup: price at or above R3 with volume spike in downtrend (mean reversion short)
            elif close[i] >= r3_aligned[i] and vol_spike[i] and ema34_1d_aligned[i] > close[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches midpoint (mean reversion target) or trend changes
            midpoint = (s3_aligned[i] + r3_aligned[i]) / 2
            if close[i] >= midpoint or ema34_1d_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price reaches midpoint or trend changes
            midpoint = (s3_aligned[i] + r3_aligned[i]) / 2
            if close[i] <= midpoint or ema34_1d_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals