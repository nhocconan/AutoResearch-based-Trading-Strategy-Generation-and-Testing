#!/usr/bin/env python3
"""
6h_TRIX_ZeroLag_Volume_Spike_Cam
Hypothesis: TRIX zero-line crossover on 6h with volume confirmation and daily trend filter. TRIX (Triple Exponential Average) filters noise and catches momentum shifts early. In bull markets, upward zero-crosses catch accelerations; in bear markets, downward crosses catch reversals. Volume spike filters false signals. Daily trend (EMA50) ensures alignment with higher timeframe. Target: 15-30 trades/year.
"""

name = "6h_TRIX_ZeroLag_Volume_Spike_Cam"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TRIX calculation (period=12)
    def ema(arr, period):
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        res = np.full(len(arr), np.nan)
        alpha = 2 / (period + 1)
        res[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            res[i] = alpha * arr[i] + (1 - alpha) * res[i-1]
        return res
    
    ema1 = ema(close, 12)
    ema2 = ema(ema1, 12)
    ema3 = ema(ema2, 12)
    trix = np.full(len(close), np.nan)
    for i in range(len(close)):
        if not np.isnan(ema3[i]) and i > 0 and not np.isnan(ema3[i-1]) and ema3[i-1] != 0:
            trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = ema(close_1d, 50)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily volume SMA20 for volume spike detection
    volume_1d = df_1d['volume'].values
    vol_sma_20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma_20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma_20_1d[i] = (vol_sma_20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 36  # warmup for TRIX (12*3) + buffer
    
    for i in range(start_idx, n):
        if np.isnan(trix[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 2.0x average 6h volume (approx from daily)
        vol_6h_approx = vol_sma_20_1d_aligned[i] / 4.0  # 24h/6h = 4
        volume_confirm = volume[i] > 2.0 * vol_6h_approx
        
        if position == 0:
            # Long: TRIX crosses above zero with uptrend and volume
            if trix[i] > 0 and trix[i-1] <= 0 and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with downtrend and volume
            elif trix[i] < 0 and trix[i-1] >= 0 and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below zero (momentum fade)
            if trix[i] < 0 and trix[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above zero (momentum fade)
            if trix[i] > 0 and trix[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals