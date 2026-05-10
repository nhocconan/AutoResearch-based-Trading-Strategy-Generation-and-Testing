#!/usr/bin/env python3
"""
4h_Trix_Pivot_Cross_1dTrend_Volume
Hypothesis: TRIX (triple exponential average) crossing its signal line (9-period EMA) combined with
1d EMA34 trend filter and volume confirmation. TRIX filters noise and captures momentum shifts.
In trending markets, TRIX signal line crossovers persist; in ranging markets, they fade.
Volume confirmation filters weak signals. Works in both bull (bullish crosses) and bear (bearish crosses).
Target: 80-160 total trades over 4 years (20-40/year).
"""

name = "4h_Trix_Pivot_Cross_1dTrend_Volume"
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
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema34_1d[i-1]
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # TRIX: 3x EMA(15) then percent change
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
    if np.any(valid1):
        start_idx = np.where(valid1)[0][0]
        for i in range(start_idx, n):
            if i == start_idx:
                ema2[i] = ema1[i]
            else:
                ema2[i] = alpha1 * ema1[i] + (1 - alpha1) * ema2[i-1]
    # EMA3
    ema3 = np.full(n, np.nan)
    valid2 = ~np.isnan(ema2)
    if np.any(valid2):
        start_idx2 = np.where(valid2)[0][0]
        for i in range(start_idx2, n):
            if i == start_idx2:
                ema3[i] = ema2[i]
            else:
                ema3[i] = alpha1 * ema2[i] + (1 - alpha1) * ema3[i-1]
    # TRIX = (EMA3 - prev EMA3) / prev EMA3 * 100
    trix = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(ema3[i]) and not np.isnan(ema3[i-1]) and ema3[i-1] != 0:
            trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    # TRIX signal line: 9-period EMA of TRIX
    trix_signal = np.full(n, np.nan)
    if n >= 9:
        # Find first valid TRIX
        first_valid = np.where(~np.isnan(trix))[0]
        if len(first_valid) > 0:
            start_idx = first_valid[0]
            trix_signal[start_idx] = trix[start_idx]
            alpha9 = 2 / (9 + 1)
            for i in range(start_idx + 1, n):
                if not np.isnan(trix[i]):
                    trix_signal[i] = alpha9 * trix[i] + (1 - alpha9) * trix_signal[i-1]
                else:
                    trix_signal[i] = trix_signal[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 15+15+15+9)  # warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or np.isnan(trix[i]) or np.isnan(trix_signal[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume approximation: 4h volume from 1d (24h/4h = 6)
        vol_4h_approx = vol_sma20_1d_aligned[i] / 6.0
        volume_confirm = volume[i] > 1.5 * vol_4h_approx
        
        if position == 0:
            # Long: TRIX crosses above signal line with uptrend and volume
            if trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1] and close[i] > ema34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal line with downtrend and volume
            elif trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1] and close[i] < ema34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below signal line or trend reversal
            if trix[i] < trix_signal[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above signal line or trend reversal
            if trix[i] > trix_signal[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals