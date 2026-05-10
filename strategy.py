#!/usr/bin/env python3
"""
4h_TRIX_ZeroLag_Volume_Spike_Cam
Hypothesis: TRIX(12) zero-lag with volume spike and 1d trend filter captures momentum bursts in both bull and bear markets.
Uses zero-lag filtering to reduce lag, volume confirmation for conviction, and 1d EMA for trend alignment.
Targets 30-40 trades/year with discrete sizing to minimize fee drag.
"""

name = "4h_TRIX_ZeroLag_Volume_Spike_Cam"
timeframe = "4h"
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
    
    # Calculate TRIX(12) with zero-lag using EMA(12) triple
    ema1 = np.full(n, np.nan)
    ema2 = np.full(n, np.nan)
    ema3 = np.full(n, np.nan)
    if n >= 12:
        alpha = 2 / (12 + 1)
        # First EMA
        ema1[11] = np.mean(close[:12])
        for i in range(12, n):
            ema1[i] = alpha * close[i] + (1 - alpha) * ema1[i-1]
        # Second EMA of first EMA
        ema2[23] = np.mean(ema1[12:24])
        for i in range(24, n):
            ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i-1]
        # Third EMA of second EMA
        ema3[35] = np.mean(ema2[24:36])
        for i in range(36, n):
            ema3[i] = alpha * ema2[i] + (1 - alpha) * ema3[i-1]
        # TRIX = (EMA3 - previous EMA3) / previous EMA3 * 100
        trix = np.full(n, np.nan)
        for i in range(37, n):
            if ema3[i-1] != 0:
                trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    
    # Zero-lag TRIX: TRIX + (TRIX - delayed TRIX) where delayed = TRIX lagged by half cycle
    # Approximate zero-lag by using current TRIX and TRIX from 6 periods ago
    trix_zl = np.full(n, np.nan)
    if n >= 43:
        for i in range(43, n):
            delayed_trix = trix[i-6] if not np.isnan(trix[i-6]) else 0
            trix_zl[i] = 2 * trix[i] - delayed_trix
    
    # Volume spike: current volume > 2.0 x volume EMA(20)
    vol_ema = np.full(n, np.nan)
    if n >= 20:
        alpha_vol = 2 / (20 + 1)
        vol_ema[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_ema[i] = alpha_vol * volume[i] + (1 - alpha_vol) * vol_ema[i-1]
    vol_spike = volume > 2.0 * vol_ema
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        alpha_1d = 2 / (34 + 1)
        ema_34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = alpha_1d * close_1d[i] + (1 - alpha_1d) * ema_34_1d[i-1]
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(44, 21)  # Ensure TRIX zero-lag and volume EMA are ready
    
    for i in range(start_idx, n):
        if np.isnan(trix_zl[i]) or np.isnan(ema_34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_spike_now = vol_spike[i] if i < len(vol_spike) else False
        
        if position == 0:
            # Long: Zero-lag TRIX crosses above zero with volume spike and uptrend
            if trix_zl[i] > 0 and trix_zl[i-1] <= 0 and vol_spike_now and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Zero-lag TRIX crosses below zero with volume spike and downtrend
            elif trix_zl[i] < 0 and trix_zl[i-1] >= 0 and vol_spike_now and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Zero-lag TRIX crosses below zero
            if trix_zl[i] < 0 and trix_zl[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Zero-lag TRIX crosses above zero
            if trix_zl[i] > 0 and trix_zl[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals