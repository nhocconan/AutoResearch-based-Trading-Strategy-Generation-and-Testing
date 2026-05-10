#!/usr/bin/env python3
"""
12h_4hMA50_Trend_Filter_Camarilla_R3S3_Breakout
Hypothesis: Breakouts of daily Camarilla R3/S3 levels in direction of 4h EMA50 trend with volume confirmation.
Works in bull/bear by following 4h trend. Target: 15-25 trades/year.
"""

name = "12h_4hMA50_Trend_Filter_Camarilla_R3S3_Breakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema_50_4h[49] = np.mean(close_4h[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_4h)):
            ema_50_4h[i] = alpha * close_4h[i] + (1 - alpha) * ema_50_4h[i-1]
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R3, S3) from previous day
    camarilla_R3 = np.full(len(close_1d), np.nan)
    camarilla_S3 = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        # Previous day's range
        range_1d = high_1d[i-1] - low_1d[i-1]
        camarilla_R3[i] = close_1d[i-1] + range_1d * 1.1 / 4
        camarilla_S3[i] = close_1d[i-1] - range_1d * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Volume spike: current volume > 2.0x average volume (30-period)
    vol_sma = np.full(n, np.nan)
    for i in range(30, n):
        vol_sma[i] = np.mean(volume[i-30:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50)  # volume SMA + EMA warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 2.0 * vol_sma[i]
        
        if position == 0:
            # Long: Close above R3 and above 4h EMA50
            if close[i] > camarilla_R3_aligned[i] and close[i] > ema_50_4h_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Close below S3 and below 4h EMA50
            elif close[i] < camarilla_S3_aligned[i] and close[i] < ema_50_4h_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below 4h EMA50
            if close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above 4h EMA50
            if close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals