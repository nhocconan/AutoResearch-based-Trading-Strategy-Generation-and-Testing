#!/usr/bin/env python3
"""
12h_KAMA_Direction_1wTrendFilter_Volume_Confirmation
Hypothesis: Uses KAMA (Kaufman Adaptive Moving Average) on 12h to capture trend direction, filtered by 1-week trend (price > 50-period EMA) and volume confirmation (volume > 1.5x 20-period average). Designed for low-frequency, high-conviction trades in both bull and bear markets. Targets 15-30 trades/year to minimize fee drag.
"""

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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get 12h data for KAMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = np.zeros_like(close_1w)
    ema_50_1w[:] = np.nan
    if len(close_1w) >= 50:
        k = 2 / (50 + 1)
        ema_50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = close_1w[i] * k + ema_50_1w[i-1] * (1 - k)
    
    # Align 1w EMA50 to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate KAMA on 12h
    close_12h = df_12h['close'].values
    kama = np.zeros_like(close_12h)
    kama[:] = np.nan
    
    if len(close_12h) >= 20:
        # Efficiency ratio
        change = np.abs(np.diff(close_12h, 10))  # 10-period change
        volatility = np.sum(np.abs(np.diff(close_12h)), axis=1)  # 10-period volatility
        er = np.zeros_like(close_12h)
        er[:10] = np.nan
        er[10:] = change[10:] / volatility[10:]
        # Avoid division by zero
        er = np.where(volatility[10:] == 0, 0, er)
        
        # Smoothing constants
        sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
        
        # Initialize KAMA
        kama[19] = np.mean(close_12h[:20])
        for i in range(20, len(close_12h)):
            kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # Align KAMA to 12h timeframe (already aligned, but keep for consistency)
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 30  # Warmup for KAMA and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price > KAMA, price > 1w EMA50, volume spike
            if close[i] > kama_aligned[i] and close[i] > ema_50_1w_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price < KAMA, price < 1w EMA50, volume spike
            elif close[i] < kama_aligned[i] and close[i] < ema_50_1w_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit: price crosses below KAMA or trend fails
            if close[i] < kama_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above KAMA or trend fails
            if close[i] > kama_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Direction_1wTrendFilter_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0