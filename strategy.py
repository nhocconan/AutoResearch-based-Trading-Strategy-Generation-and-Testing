#!/usr/bin/env python3
"""
12h_TRIX_ZeroLag_Volume_Spike_1wTrend
Hypothesis: TRIX zero-line cross in direction of weekly trend with volume confirmation.
TRIX filters noise and captures momentum; weekly trend ensures alignment with higher timeframe.
Works in bull/bear by following weekly trend. Target: 15-25 trades/year.
"""

name = "12h_TRIX_ZeroLag_Volume_Spike_1wTrend"
timeframe = "12h"
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
    
    # TRIX calculation (15-period EMA applied 3 times)
    # EMA1
    ema1 = np.full(n, np.nan)
    if n >= 15:
        ema1[14] = np.mean(close[:15])
        alpha1 = 2 / (15 + 1)
        for i in range(15, n):
            ema1[i] = alpha1 * close[i] + (1 - alpha1) * ema1[i-1]
    # EMA2
    ema2 = np.full(n, np.nan)
    valid1 = ~np.isnan(ema1)
    if np.sum(valid1) >= 15:
        start = np.where(valid1)[0][14]
        ema2[start] = np.mean(ema1[start-14:start+1])
        alpha2 = 2 / (15 + 1)
        for i in range(start+1, n):
            if np.isnan(ema1[i]):
                ema2[i] = np.nan
            else:
                ema2[i] = alpha2 * ema1[i] + (1 - alpha2) * ema2[i-1]
    # EMA3
    ema3 = np.full(n, np.nan)
    valid2 = ~np.isnan(ema2)
    if np.sum(valid2) >= 15:
        start = np.where(valid2)[0][14]
        ema3[start] = np.mean(ema2[start-14:start+1])
        alpha3 = 2 / (15 + 1)
        for i in range(start+1, n):
            if np.isnan(ema2[i]):
                ema3[i] = np.nan
            else:
                ema3[i] = alpha3 * ema2[i] + (1 - alpha3) * ema3[i-1]
    # TRIX = (EMA3 - prev EMA3) / prev EMA3 * 100
    trix = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(ema3[i]) and not np.isnan(ema3[i-1]) and ema3[i-1] != 0:
            trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    
    # Volume spike: current volume > 2.0x average volume (30-period)
    vol_sma = np.full(n, np.nan)
    for i in range(30, n):
        vol_sma[i] = np.mean(volume[i-30:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 45)  # volume SMA + TRIX warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(trix[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 2.0 * vol_sma[i]
        
        if position == 0:
            # Long: TRIX crosses above zero and above weekly EMA34
            if trix[i] > 0 and trix[i-1] <= 0 and close[i] > ema_34_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero and below weekly EMA34
            elif trix[i] < 0 and trix[i-1] >= 0 and close[i] < ema_34_1w_aligned[i] and volume_confirm:
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