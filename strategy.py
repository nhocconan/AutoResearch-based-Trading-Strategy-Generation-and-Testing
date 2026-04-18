#!/usr/bin/env python3
"""
1d Volume-Weighted RSI with Weekly Supertrend Filter
Mean reversion on daily timeframe using volume-weighted RSI filtered by weekly Supertrend trend.
Designed to work in both bull and bear markets by fading extremes in range-bound conditions
while respecting the weekly trend direction.
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
    
    # Get 1d data for VW-RSI calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Volume-Weighted RSI(14) on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Volume-weighted gain/loss
    vol_gain = gain * volume_1d
    vol_loss = loss * volume_1d
    
    # Wilder smoothing with volume weighting
    avg_vg = np.zeros_like(close_1d)
    avg_vl = np.zeros_like(close_1d)
    avg_vg[0] = vol_gain[0]
    avg_vl[0] = vol_loss[0]
    
    for i in range(1, len(close_1d)):
        avg_vg[i] = (avg_vg[i-1] * 13 + vol_gain[i]) / 14
        avg_vl[i] = (avg_vl[i-1] * 13 + vol_loss[i]) / 14
    
    rs = np.where(avg_vl != 0, avg_vg / avg_vl, 100)
    vw_rsi = 100 - (100 / (1 + rs))
    
    # Get 1w data for Supertrend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR(10) for Supertrend
    tr1 = np.abs(np.diff(high_1w, prepend=high_1w[0]))
    tr2 = np.abs(np.diff(low_1w, prepend=low_1w[0]))
    tr3 = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.zeros_like(close_1w)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 9 + tr[i]) / 10
    
    # Supertrend calculation
    factor = 3.0
    hl2 = (high_1w + low_1w) / 2
    upper = hl2 + factor * atr
    lower = hl2 - factor * atr
    
    supertrend = np.zeros_like(close_1w)
    dir = np.ones_like(close_1w, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper[0]
    dir[0] = 1
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > supertrend[i-1]:
            dir[i] = 1
        else:
            dir[i] = -1
        
        if dir[i] == 1:
            supertrend[i] = max(lower[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper[i], supertrend[i-1])
    
    # Align indicators to 1d timeframe
    vw_rsi_aligned = align_htf_to_ltf(prices, df_1d, vw_rsi)
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    dir_aligned = align_htf_to_ltf(prices, df_1w, dir)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(vw_rsi_aligned[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(dir_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi = vw_rsi_aligned[i]
        supertrend_val = supertrend_aligned[i]
        trend_dir = dir_aligned[i]
        
        if position == 0:
            # Long: oversold VW-RSI in uptrend
            if rsi < 30 and trend_dir == 1:
                signals[i] = 0.25
                position = 1
            # Short: overbought VW-RSI in downtrend
            elif rsi > 70 and trend_dir == -1:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses above 50 or trend changes
            if rsi > 50 or trend_dir == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses below 50 or trend changes
            if rsi < 50 or trend_dir == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_VolumeWeightedRSI_WeeklySupertrend"
timeframe = "1d"
leverage = 1.0