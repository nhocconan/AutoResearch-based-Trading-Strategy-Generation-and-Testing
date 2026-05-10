#!/usr/bin/env python3
"""
6h_TRIX_ZeroLag_Volume_Spike_Cam
Hypothesis: TRIX (zero-lag) combined with Camarilla pivot levels and volume spikes captures momentum in both bull and bear markets. TRIX zero-lag reduces lag for timely entries, Camarilla R3/S3 levels provide institutional support/resistance, and volume spikes confirm institutional participation. Trend filter from 12h EMA50 avoids counter-trend trades.
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
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TRIX zero-lag (12-period)
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        alpha = 2 / (period + 1)
        result = np.full_like(arr, np.nan, dtype=float)
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
        return result
    
    ema1 = ema(close, 12)
    ema2 = ema(ema1, 12)
    ema3 = ema(ema2, 12)
    # Zero-lag TRIX: 3*ema3 - 3*ema2 + ema1
    trix_zl = 3 * ema3 - 3 * ema2 + ema1
    # Normalize by price to get percentage
    trix_zl_pct = (trix_zl / close) * 100
    
    # Camarilla levels from 1d
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla for each 1d bar
    camarilla_r4 = np.full_like(close_1d, np.nan)
    camarilla_r3 = np.full_like(close_1d, np.nan)
    camarilla_s3 = np.full_like(close_1d, np.nan)
    camarilla_s4 = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if i == 0 or np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]):
            continue
        range_ = high_1d[i] - low_1d[i]
        camarilla_r4[i] = close_1d[i] + range_ * 1.1 / 2
        camarilla_r3[i] = close_1d[i] + range_ * 1.1 / 4
        camarilla_s3[i] = close_1d[i] - range_ * 1.1 / 4
        camarilla_s4[i] = close_1d[i] - range_ * 1.1 / 2
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 12h EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = ema(close_12h, 50)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 2.0x average volume (20-period)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(36, 20)  # TRIX needs 36 bars (3*12), vol needs 20
    
    for i in range(start_idx, n):
        if np.isnan(trix_zl_pct[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 2.0 * vol_sma[i]
        
        if position == 0:
            # Long: TRIX zero-lag turning up, above Camarilla R3, above 12h EMA50
            if trix_zl_pct[i] > trix_zl_pct[i-1] and trix_zl_pct[i] > 0 and close[i] > camarilla_r3_aligned[i] and close[i] > ema_50_12h_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: TRIX zero-lag turning down, below Camarilla S3, below 12h EMA50
            elif trix_zl_pct[i] < trix_zl_pct[i-1] and trix_zl_pct[i] < 0 and close[i] < camarilla_s3_aligned[i] and close[i] < ema_50_12h_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX zero-lag turns down OR price breaks below Camarilla S3
            if trix_zl_pct[i] < trix_zl_pct[i-1] or close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX zero-lag turns up OR price breaks above Camarilla R3
            if trix_zl_pct[i] > trix_zl_pct[i-1] or close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals