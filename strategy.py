#!/usr/bin/env python3
"""
4h_TRIX_ZeroLine_1dTrend_Volume
Hypothesis: TRIX zero-line cross in direction of 1d EMA134 trend with volume confirmation.
Works in bull/bear by following 1d trend. Target: 15-35 trades/year.
"""

name = "4h_TRIX_ZeroLine_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TRIX (15-period EMA of EMA of EMA, then ROC)
    def ema(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        alpha = 2 / (period + 1)
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
        return result
    
    ema1 = ema(close, 15)
    ema2 = ema(ema1, 15)
    ema3 = ema(ema2, 15)
    trix = np.full_like(close, np.nan, dtype=float)
    for i in range(1, len(ema3)):
        if not np.isnan(ema3[i]) and not np.isnan(ema3[i-1]):
            trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    
    # 1d EMA134 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_134_1d = ema(close_1d, 134)
    ema_134_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_134_1d)
    
    # Volume spike: current volume > 1.5x average volume (20-period)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # TRIX + volume warmup
    
    for i in range(start_idx, n):
        if np.isnan(trix[i]) or np.isnan(ema_134_1d_aligned[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 0:
            # Long: TRIX crosses above zero and above 1d EMA134
            if trix[i] > 0 and trix[i-1] <= 0 and close[i] > ema_134_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero and below 1d EMA134
            elif trix[i] < 0 and trix[i-1] >= 0 and close[i] < ema_134_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below zero
            if trix[i] < 0 and trix[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above zero
            if trix[i] > 0 and trix[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals