#!/usr/bin/env python3
# 4h_Trix_ZeroCross_1dTrend_VolumeConfirm
# Hypothesis: TRIX (12) zero cross signals momentum shifts. Long when TRIX crosses above zero with 1d uptrend (close > EMA50) and volume > 1.5x average.
# Short when TRIX crosses below zero with 1d downtrend (close < EMA50) and volume confirmation. Uses TRIX histogram for smoother zero-cross detection.
# Designed for 20-40 trades per year on 4h timeframe. Works in bull markets via momentum continuation and in bear via momentum reversals.

name = "4h_Trix_ZeroCross_1dTrend_VolumeConfirm"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50
    
    # Align 1d EMA to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate TRIX (12,12,12) - triple smoothed EMA of ROC
    # TRIX = EMA(EMA(EMA(ROC, 12), 12), 12)
    roc = np.full_like(close, np.nan)
    roc[12:] = (close[12:] - close[:-12]) / close[:-12] * 100
    
    # First EMA
    ema1 = np.full_like(roc, np.nan)
    if len(roc) >= 12:
        ema1[11] = np.mean(roc[0:12])
        for i in range(12, len(roc)):
            ema1[i] = (roc[i] * 2 + ema1[i-1] * 10) / 12
    
    # Second EMA
    ema2 = np.full_like(ema1, np.nan)
    if len(ema1) >= 12:
        ema2[11] = np.mean(ema1[0:12])
        for i in range(12, len(ema1)):
            ema2[i] = (ema1[i] * 2 + ema2[i-1] * 10) / 12
    
    # Third EMA (TRIX)
    ema3 = np.full_like(ema2, np.nan)
    if len(ema2) >= 12:
        ema3[11] = np.mean(ema2[0:12])
        for i in range(12, len(ema2)):
            ema3[i] = (ema2[i] * 2 + ema3[i-1] * 10) / 12
    
    # TRIX histogram for zero-cross detection (smoother than raw TRIX)
    trix_hist = ema3
    
    # Volume filter: 4h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure TRIX and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(trix_hist[i]) or np.isnan(trix_hist[i-1]) or \
           np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: TRIX crosses above zero AND volume confirmation AND bullish trend (close > EMA50)
            if trix_hist[i-1] <= 0 and trix_hist[i] > 0 and volume_ratio[i] > 1.5 and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero AND volume confirmation AND bearish trend (close < EMA50)
            elif trix_hist[i-1] >= 0 and trix_hist[i] < 0 and volume_ratio[i] > 1.5 and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero (momentum reversal) or trend turns bearish
            if trix_hist[i-1] > 0 and trix_hist[i] <= 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero (momentum reversal) or trend turns bullish
            if trix_hist[i-1] < 0 and trix_hist[i] >= 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals