#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_Volume
Hypothesis: Camarilla R3/S3 breakout in direction of 1w EMA34 trend with volume confirmation.
Works in bull/bear by following weekly trend. Target: 15-25 trades/year.
"""

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
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
    
    # 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_34_1w[i-1]
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
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
    
    start_idx = max(30, 34)  # volume SMA + EMA warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 2.0 * vol_sma[i]
        
        if position == 0:
            # Long: Close above R3 and above weekly EMA34
            if close[i] > camarilla_R3_aligned[i] and close[i] > ema_34_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Close below S3 and below weekly EMA34
            elif close[i] < camarilla_S3_aligned[i] and close[i] < ema_34_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below weekly EMA34
            if close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above weekly EMA34
            if close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals