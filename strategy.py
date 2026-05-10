#!/usr/bin/env python3
"""
12h_TRIX_ZeroCross_VolumeSpike
Hypothesis: Use TRIX (triple smoothed EMA) on 1d for momentum, with zero-cross signals.
Enter long when TRIX crosses above zero with volume spike (>2x 20-period average).
Enter short when TRIX crosses below zero with volume spike.
Use 1w trend filter: only take long when price > 200 EMA on 1w, short when price < 200 EMA on 1w.
Targets 50-120 trades over 4 years (12-30/year) to minimize fee drag.
TRIX reduces whipsaw vs single EMA, volume confirms momentum, 1w trend filter avoids counter-trend trades.
Works in bull (rides momentum with trend) and bear (captures reversal spikes with trend filter).
"""

name = "12h_TRIX_ZeroCross_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d TRIX (15-period triple EMA)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate TRIX: EMA(EMA(EMA(close, 15), 15), 15)
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        res = np.full_like(arr, np.nan, dtype=float)
        alpha = 2 / (period + 1)
        res[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            res[i] = alpha * arr[i] + (1 - alpha) * res[i-1]
        return res
    
    ema1 = ema(close_1d, 15)
    ema2 = ema(ema1, 15)
    ema3 = ema(ema2, 15)
    trix = np.full_like(close_1d, np.nan, dtype=float)
    # TRIX = 100 * (ema3 - previous ema3) / previous ema3
    for i in range(1, len(ema3)):
        if not np.isnan(ema3[i]) and not np.isnan(ema3[i-1]) and ema3[i-1] != 0:
            trix[i] = 100 * (ema3[i] - ema3[i-1]) / ema3[i-1]
    
    # 1d volume SMA20 for volume confirmation
    vol_sma20_1d = np.full_like(vol_1d, np.nan, dtype=float)
    if len(vol_1d) >= 20:
        vol_sma20_1d[19] = np.mean(vol_1d[:20])
        for i in range(20, len(vol_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + vol_1d[i]) / 20
    
    # 1w EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema200_1w = np.full_like(close_1w, np.nan, dtype=float)
    if len(close_1w) >= 200:
        ema200_1w[199] = np.mean(close_1w[:200])
        alpha = 2 / (200 + 1)
        for i in range(200, len(close_1w)):
            ema200_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema200_1w[i-1]
    
    # Align all to 12h timeframe
    trix_12h = align_htf_to_ltf(prices, df_1d, trix)
    vol_sma20_12h = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    ema200_12h = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 1)  # Need TRIX and volume
    
    for i in range(start_idx, n):
        if np.isnan(trix_12h[i]) or np.isnan(vol_sma20_12h[i]) or np.isnan(ema200_12h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 2x average 1d volume (scaled to 12h)
        # 2x 12h periods in 1d, so scale factor is 2
        vol_12h_avg = vol_sma20_12h[i] * 2
        volume_confirm = volume[i] > 2.0 * vol_12h_avg
        
        # TRIX zero-cross detection
        trix_prev = trix_12h[i-1] if i > 0 else 0
        trix_cross_up = trix_prev <= 0 and trix_12h[i] > 0
        trix_cross_down = trix_prev >= 0 and trix_12h[i] < 0
        
        if position == 0:
            # Long: TRIX crosses up with volume and price above 1w EMA200
            if trix_cross_up and volume_confirm and close[i] > ema200_12h[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses down with volume and price below 1w EMA200
            elif trix_cross_down and volume_confirm and close[i] < ema200_12h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses down or trend fails
            if trix_cross_down or close[i] < ema200_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses up or trend fails
            if trix_cross_up or close[i] > ema200_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals