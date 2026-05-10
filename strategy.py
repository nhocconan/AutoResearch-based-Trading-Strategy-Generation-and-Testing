#!/usr/bin/env python3
"""
4h_TRIX_ZeroLag_Volume_Spike_12hTrend
Hypothesis: TRIX zero-cross on 4h with volume spike and 12h EMA trend filter.
Works in bull/bear by following 12h trend. Target: 20-40 trades/year.
"""

name = "4h_TRIX_ZeroLag_Volume_Spike_12hTrend"
timeframe = "4h"
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
    high = prices['high'].values
    low = prices['low'].values
    
    # 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 34:
        ema_34_12h[33] = np.mean(close_12h[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_12h)):
            ema_34_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_34_12h[i-1]
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # TRIX zero-lag on 4h (12-period)
    ema1 = np.full(n, np.nan)
    ema2 = np.full(n, np.nan)
    ema3 = np.full(n, np.nan)
    if n >= 12:
        # First EMA
        ema1[11] = np.mean(close[:12])
        alpha = 2 / (12 + 1)
        for i in range(12, n):
            ema1[i] = alpha * close[i] + (1 - alpha) * ema1[i-1]
        # Second EMA of EMA1
        ema2[23] = np.mean(ema1[12:24])  # Need 12 values of ema1
        for i in range(24, n):
            ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i-1]
        # Third EMA of EMA2
        ema3[35] = np.mean(ema2[24:36])  # Need 12 values of ema2
        for i in range(36, n):
            ema3[i] = alpha * ema2[i] + (1 - alpha) * ema3[i-1]
        # TRIX = (EMA3 - previous EMA3) / previous EMA3 * 100
        trix = np.full(n, np.nan)
        for i in range(37, n):
            if ema3[i-1] != 0:
                trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    
    # Volume spike: current volume > 1.5x average volume (20-period)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(36, 20, 34)  # TRIX + volume SMA + EMA warmup
    
    for i in range(start_idx, n):
        if np.isnan(trix[i]) or np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 0:
            # Long: TRIX crosses above zero and above 12h EMA34
            if trix[i] > 0 and trix[i-1] <= 0 and close[i] > ema_34_12h_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero and below 12h EMA34
            elif trix[i] < 0 and trix[i-1] >= 0 and close[i] < ema_34_12h_aligned[i] and volume_confirm:
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