#!/usr/bin/env python3
"""
12h_1d_Volume_Weighted_Camarilla_Breakout
Hypothesis: 12h price break above/below 1d Camarilla R3/S3 levels with volume confirmation
and 1d trend filter (EMA34). Works in bull/bear by following 1d trend. Target: 15-30 trades/year.
"""

name = "12h_1d_Volume_Weighted_Camarilla_Breakout"
timeframe = "12h"
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
    volume = volumes = prices['volume'].values  # Fix: volumes -> volume
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_34_1d[i-1]
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d Camarilla R3 and S3 levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_r3 = np.zeros(len(close_1d))
    camarilla_s3 = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_r3[i] = close_1d[0]
            camarilla_s3[i] = close_1d[0]
        else:
            camarilla_r3[i] = close_1d[i-1] + (high_1d[i-1] - low_1d[i-1]) * 1.1666
            camarilla_s3[i] = close_1d[i-1] - (high_1d[i-1] - low_1d[i-1]) * 1.1666
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume spike: current volume > 2.0x average volume (30-period)
    vol_sma = np.full(n, np.nan)
    for i in range(30, n):
        vol_sma[i] = np.mean(volume[i-30:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 30)  # EMA + volume warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 2.0 * vol_sma[i]
        
        if position == 0:
            # Long: Break above R3 and above 1d EMA34
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 and below 1d EMA34
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below 1d EMA34
            if close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above 1d EMA34
            if close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals