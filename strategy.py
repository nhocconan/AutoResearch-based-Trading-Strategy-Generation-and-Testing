#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Buy when price breaks above Camarilla R3 with above-average volume and price above 1-day EMA34; short when price breaks below S3 with volume and below EMA34. Exit when price crosses back below/above EMA34. Works in bull/bear by following daily trend. Target: 20-30 trades/year.
"""

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # 1-day EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_34_1d[i-1]
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily data for Camarilla pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels (R3, S3) from previous day
    camarilla_R3 = np.full(len(close_1d), np.nan)
    camarilla_S3 = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        # Previous day's range
        range_1d = high_1d[i-1] - low_1d[i-1]
        camarilla_R3[i] = close_1d[i-1] + range_1d * 1.1 / 4
        camarilla_S3[i] = close_1d[i-1] - range_1d * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Volume spike: current volume > 1.5x average volume (20-period)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # volume SMA + EMA warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 0:
            # Long: Close above R3 and above daily EMA34
            if close[i] > camarilla_R3_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Close below S3 and below daily EMA34
            elif close[i] < camarilla_S3_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below daily EMA34
            if close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above daily EMA34
            if close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals