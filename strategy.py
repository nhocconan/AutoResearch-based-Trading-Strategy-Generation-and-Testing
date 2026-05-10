#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1wTrend_Volume
Hypothesis: 4h Camarilla R3/S3 level breakout in direction of weekly EMA10 trend with volume confirmation.
Uses weekly trend for multi-timeframe alignment, works in bull/bear by following higher timeframe trend.
Target: 20-40 trades/year to stay under fee drag limits.
"""

name = "4h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "4h"
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
    
    # Weekly EMA10 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_10_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 10:
        ema_10_1w[9] = np.mean(close_1w[:10])
        alpha = 2 / (10 + 1)
        for i in range(10, len(close_1w)):
            ema_10_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_10_1w[i-1]
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R3, S3) from previous day
    r3 = np.full(len(close_1d), np.nan)
    s3 = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        # Previous day's range
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        if range_val > 0:
            r3[i] = prev_close + range_val * 1.1 / 4
            s3[i] = prev_close - range_val * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: current volume > 1.5x average volume (20-period)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)  # Volume + weekly EMA warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_10_1w_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 0:
            # Long: Break above R3 and above weekly EMA10
            if close[i] > r3_aligned[i] and close[i] > ema_10_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 and below weekly EMA10
            elif close[i] < s3_aligned[i] and close[i] < ema_10_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below weekly EMA10
            if close[i] < ema_10_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above weekly EMA10
            if close[i] > ema_10_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals