#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_Volume_v2
Hypothesis: Refined version with improved win rate and reduced trade count by adding 
ATR volatility filter and stricter volume confirmation. Uses 1w EMA34 trend filter,
Camarilla R3/S3 from previous day, and requires volatility expansion (ATR ratio > 1.2).
Designed for 12-37 trades/year target range.
"""

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume_v2"
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
    
    # Daily data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on daily
    atr_14_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 14:
        tr = np.zeros(len(close_1d))
        tr[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr[i] = max(high_1d[i] - low_1d[i], 
                       abs(high_1d[i] - close_1d[i-1]),
                       abs(low_1d[i] - close_1d[i-1]))
        # Wilder's smoothing
        atr_14_1d[13] = np.mean(tr[1:14])
        for i in range(14, len(close_1d)):
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate ATR ratio (current ATR / 20-period average) for volatility filter
    atr_ratio_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:  # 14 + 20
        atr_ma_20 = np.full(len(close_1d), np.nan)
        for i in range(33, len(close_1d)):
            atr_ma_20[i] = np.mean(atr_14_1d[i-19:i+1])
        for i in range(33, len(close_1d)):
            if atr_ma_20[i] > 0:
                atr_ratio_1d[i] = atr_14_1d[i] / atr_ma_20[i]
    
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate Camarilla levels (R3, S3) from previous day
    camarilla_R3 = np.full(len(close_1d), np.nan)
    camarilla_S3 = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        range_1d = high_1d[i-1] - low_1d[i-1]
        camarilla_R3[i] = close_1d[i-1] + range_1d * 1.1 / 4
        camarilla_S3[i] = close_1d[i-1] - range_1d * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Volume spike: current volume > 2.5x average volume (20-period) - stricter
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(33, 34)  # ATR ratio (33) + EMA warmup (34)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(vol_sma[i]) or 
            np.isnan(atr_ratio_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: ATR ratio > 1.2 (expanding volatility)
        vol_filter = atr_ratio_1d_aligned[i] > 1.2
        
        # Volume confirmation: current volume > 2.5x average
        volume_confirm = volume[i] > 2.5 * vol_sma[i]
        
        if position == 0:
            # Long: Close above R3, above weekly EMA34, with vol and volume confirmation
            if (close[i] > camarilla_R3_aligned[i] and close[i] > ema_34_1w_aligned[i] and 
                vol_filter and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Close below S3, below weekly EMA34, with vol and volume confirmation
            elif (close[i] < camarilla_S3_aligned[i] and close[i] < ema_34_1w_aligned[i] and 
                  vol_filter and volume_confirm):
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