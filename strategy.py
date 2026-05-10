#!/usr/bin/env python3
"""
6h_TrixZerosignal_VolumeSpike_12hTrend
Hypothesis: Use zero-lag Trix momentum with volume spikes filtered by 12h EMA trend direction.
Works in bull/bear by following 12h trend while capturing momentum bursts. Target: 10-30 trades/year.
"""

name = "6h_TrixZerosignal_VolumeSpike_12hTrend"
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
    
    # Zero-lag Trix (TRIX with EMA smoothing to reduce lag)
    # EMA1
    ema1 = np.full(n, np.nan)
    if n >= 15:
        ema1[14] = np.mean(close[:15])
        alpha1 = 2 / (15 + 1)
        for i in range(15, n):
            ema1[i] = alpha1 * close[i] + (1 - alpha1) * ema1[i-1]
    # EMA2 of EMA1
    ema2 = np.full(n, np.nan)
    if n >= 15:
        ema2[14] = np.mean(ema1[:15])
        alpha2 = 2 / (15 + 1)
        for i in range(15, n):
            ema2[i] = alpha2 * ema1[i] + (1 - alpha2) * ema2[i-1]
    # EMA3 of EMA2
    ema3 = np.full(n, np.nan)
    if n >= 15:
        ema3[14] = np.mean(ema2[:15])
        alpha3 = 2 / (15 + 1)
        for i in range(15, n):
            ema3[i] = alpha3 * ema2[i] + (1 - alpha3) * ema3[i-1]
    # TRIX = 100 * (EMA3 - prev EMA3) / prev EMA3
    trix = np.full(n, np.nan)
    for i in range(16, n):
        if ema3[i-1] != 0:
            trix[i] = 100 * (ema3[i] - ema3[i-1]) / ema3[i-1]
    # Signal line: EMA of TRIX
    trix_signal = np.full(n, np.nan)
    if n >= 9:
        trix_signal[8] = np.mean(trix[9:17]) if n >= 17 else np.nan
        alpha_s = 2 / (9 + 1)
        for i in range(9, n):
            if not np.isnan(trix[i]):
                trix_signal[i] = alpha_s * trix[i] + (1 - alpha_s) * trix_signal[i-1]
    # Zero-lag component: 2*TRIX - signal
    ztrix = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(trix[i]) and not np.isnan(trix_signal[i]):
            ztrix[i] = 2 * trix[i] - trix_signal[i]
    
    # Volume spike: current volume > 2.0x average volume (20-period)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 17, 9)  # EMA + volume + TRIX warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_12h_aligned[i]) or np.isnan(ztrix[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 2.0 * vol_sma[i]
        
        if position == 0:
            # Long: Positive zero-lag Trix and above 12h EMA34 with volume
            if ztrix[i] > 0.1 and close[i] > ema_34_12h_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Negative zero-lag Trix and below 12h EMA34 with volume
            elif ztrix[i] < -0.1 and close[i] < ema_34_12h_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Zero-lag Trix turns negative or price below EMA
            if ztrix[i] < -0.05 or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Zero-lag Trix turns positive or price above EMA
            if ztrix[i] > 0.05 or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals