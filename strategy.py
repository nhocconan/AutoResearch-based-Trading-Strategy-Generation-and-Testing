#!/usr/bin/env python3
# 4h_TRIX_ZeroCross_12hTrend_VolumeSpike
# Hypothesis: TRIX zero-cross with 12h trend filter and volume spike confirmation.
# TRIX captures momentum with less whipsaw than MACD. Zero-cross signals momentum shifts.
# 12h trend filter avoids counter-trend trades. Volume confirms momentum strength.
# Designed for low trade frequency (<50/year) to minimize fee drag in 2025 bear market.

name = "4h_TRIX_ZeroCross_12hTrend_VolumeSpike"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for TRIX calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate TRIX: Triple EMA of close, then 1-period percent change
    # EMA1
    ema1 = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 12:
        ema1[11] = np.mean(close_12h[0:12])
        for i in range(12, len(close_12h)):
            ema1[i] = (close_12h[i] * 2/13) + (ema1[i-1] * 11/13)
    
    # EMA2 of EMA1
    ema2 = np.full_like(close_12h, np.nan)
    valid1 = ~np.isnan(ema1)
    if np.sum(valid1) >= 12:
        start_idx = np.where(valid1)[0][11]  # 12th valid value
        ema2[start_idx] = np.mean(ema1[start_idx-11:start_idx+1])
        for i in range(start_idx+1, len(close_12h)):
            if not np.isnan(ema1[i]):
                ema2[i] = (ema1[i] * 2/13) + (ema2[i-1] * 11/13)
    
    # EMA3 of EMA2
    ema3 = np.full_like(close_12h, np.nan)
    valid2 = ~np.isnan(ema2)
    if np.sum(valid2) >= 12:
        start_idx = np.where(valid2)[0][11]  # 12th valid value
        ema3[start_idx] = np.mean(ema2[start_idx-11:start_idx+1])
        for i in range(start_idx+1, len(close_12h)):
            if not np.isnan(ema2[i]):
                ema3[i] = (ema2[i] * 2/13) + (ema3[i-1] * 11/13)
    
    # TRIX = (EMA3 - previous EMA3) / previous EMA3 * 100
    trix = np.full_like(close_12h, np.nan)
    valid3 = ~np.isnan(ema3)
    for i in range(1, len(close_12h)):
        if valid3[i] and valid3[i-1] and ema3[i-1] != 0:
            trix[i] = ((ema3[i] - ema3[i-1]) / ema3[i-1]) * 100
    
    # Align TRIX to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix)
    
    # Get 12h EMA50 for trend filter
    ema_50_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[0:50])
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = (close_12h[i] * 2/51) + (ema_50_12h[i-1] * 49/51)
    
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (volume[i] * 2/21) + (vol_ma[i-1] * 19/21)
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: TRIX crosses above zero AND uptrend (close > EMA50) AND volume spike
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero AND downtrend (close < EMA50) AND volume spike
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero OR trend reversal (close < EMA50)
            if (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0) or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero OR trend reversal (close > EMA50)
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0) or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals