#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Volume_Spice
Breakout strategy using Camarilla pivot levels from 1d with volume spike confirmation.
Long when price breaks above R3 with volume > 1.5x 20-period average.
Short when price breaks below S3 with volume > 1.5x 20-period average.
Exit when price returns to H4/L4 levels or volume drops below average.
Uses 12h EMA50 as trend filter to avoid counter-trend trades.
Target: 20-40 trades/year per symbol.
"""

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
    
    # Calculate 20-period average volume for spike detection
    vol_avg = np.full(n, np.nan)
    for i in range(19, n):
        vol_avg[i] = np.mean(volume[i-19:i+1])
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_h5 = np.zeros(len(close_1d))
    camarilla_h4 = np.zeros(len(close_1d))
    camarilla_h3 = np.zeros(len(close_1d))
    camarilla_l3 = np.zeros(len(close_1d))
    camarilla_l4 = np.zeros(len(close_1d))
    camarilla_l5 = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        if i == 0:
            # Use first available values for day 0
            camarilla_h5[i] = high_1d[i]
            camarilla_h4[i] = high_1d[i]
            camarilla_h3[i] = high_1d[i]
            camarilla_l3[i] = low_1d[i]
            camarilla_l4[i] = low_1d[i]
            camarilla_l5[i] = low_1d[i]
        else:
            # Standard Camarilla calculation using previous day's range
            rng = high_1d[i-1] - low_1d[i-1]
            camarilla_h5[i] = close_1d[i-1] + (rng * 1.500)
            camarilla_h4[i] = close_1d[i-1] + (rng * 1.250)
            camarilla_h3[i] = close_1d[i-1] + (rng * 1.166)
            camarilla_l3[i] = close_1d[i-1] - (rng * 1.166)
            camarilla_l4[i] = close_1d[i-1] - (rng * 1.250)
            camarilla_l5[i] = close_1d[i-1] - (rng * 1.500)
    
    # Align Camarilla levels to 4h timeframe
    h5_4h = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    h4_4h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_4h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    l5_4h = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    # Get 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50
    ema_12h_period = 50
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= ema_12h_period:
        ema_12h[ema_12h_period - 1] = np.mean(close_12h[:ema_12h_period])
        for i in range(ema_12h_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * (2 / (ema_12h_period + 1)) + 
                         ema_12h[i - 1] * (1 - (2 / (ema_12h_period + 1))))
    
    # Align 12h EMA50 to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need volume average and EMA
    start_idx = max(19, ema_12h_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vol_avg[i]) or np.isnan(ema_12h_aligned[i]) or
            np.isnan(h5_4h[i]) or np.isnan(h4_4h[i]) or np.isnan(h3_4h[i]) or
            np.isnan(l3_4h[i]) or np.isnan(l4_4h[i]) or np.isnan(l5_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_average = vol_avg[i]
        ema12h_val = ema_12h_aligned[i]
        
        if position == 0:
            # Long: Price breaks above H3 with volume spike and above 12h EMA50 (uptrend)
            if (price > h3_4h[i] and vol > 1.5 * vol_average and price > ema12h_val):
                signals[i] = size
                position = 1
            # Short: Price breaks below L3 with volume spike and below 12h EMA50 (downtrend)
            elif (price < l3_4h[i] and vol > 1.5 * vol_average and price < ema12h_val):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price returns to H4 or volume drops below average
            if (price <= h4_4h[i] or vol < vol_average):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Price returns to L4 or volume drops below average
            if (price >= l4_4h[i] or vol < vol_average):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_Pivot_Volume_Spice"
timeframe = "4h"
leverage = 1.0