#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla R3/S3 breakouts on 12h with 1d EMA trend filter and volume confirmation.
This strategy trades breakouts in the direction of the daily trend, using Camarilla levels
from the previous day. Works in both bull and bear markets by aligning with higher timeframe trend.
Volume confirmation filters out false breakouts. Low trade frequency reduces fee drag.
"""

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    
    # === 1d Data for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # === 1d Data for Camarilla Pivot Levels (previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R3, S3
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align to 12h timeframe
    camarilla_r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === Volume Filter (1.5x 20-period EMA on 12h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers 1d EMA and Camarilla calculation)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r3_12h[i]) or np.isnan(camarilla_s3_12h[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price closes above R3 with uptrend and volume
            if (close[i] > camarilla_r3_12h[i] and 
                close[i] > ema34_aligned[i] and 
                volume_ok[i]):
                signals[i] = 0.30
                position = 1
            # Short breakdown: price closes below S3 with downtrend and volume
            elif (close[i] < camarilla_s3_12h[i] and 
                  close[i] < ema34_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price closes below S3 (mean reversion)
            if close[i] < camarilla_s3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30  # maintain position
        elif position == -1:
            # Short exit: price closes above R3 (mean reversion)
            if close[i] > camarilla_r3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30  # maintain position
    
    return signals