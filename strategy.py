#!/usr/bin/env python3
# 4h_TRIX_ZeroCross_1dTrend_VolumeSpike
# Hypothesis: TRIX zero-cross signals on 4h timeframe filtered by 1d EMA trend and volume spikes.
# TRIX captures momentum shifts, EMA trend ensures directional bias, volume confirms institutional participation.
# Works in bull/bear markets by avoiding counter-trend trades and requiring volume confirmation for entries.

name = "4h_TRIX_ZeroCross_1dTrend_VolumeSpike"
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
    
    # Calculate TRIX on 4h close (12-period EMA of EMA of EMA, then ROC)
    if len(close) < 12:
        return np.zeros(n)
    
    # First EMA
    ema1 = np.full_like(close, np.nan)
    ema1[0] = close[0]
    for i in range(1, len(close)):
        ema1[i] = (close[i] * 2 + ema1[i-1] * 10) / 12  # 12-period EMA
    
    # Second EMA
    ema2 = np.full_like(close, np.nan)
    ema2[0] = ema1[0]
    for i in range(1, len(close)):
        ema2[i] = (ema1[i] * 2 + ema2[i-1] * 10) / 12
    
    # Third EMA
    ema3 = np.full_like(close, np.nan)
    ema3[0] = ema2[0]
    for i in range(1, len(close)):
        ema3[i] = (ema2[i] * 2 + ema3[i-1] * 10) / 12
    
    # TRIX = 100 * (EMA3 today - EMA3 yesterday) / EMA3 yesterday
    trix = np.full_like(close, np.nan)
    for i in range(1, len(close)):
        if ema3[i-1] != 0:
            trix[i] = 100 * (ema3[i] - ema3[i-1]) / ema3[i-1]
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2 + ema_34_1d[i-1] * 32) / 34
    
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (volume[i] * 2 + vol_ma[i-1] * 18) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(1, 20)  # Ensure TRIX and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: TRIX crosses above zero AND uptrend (price > EMA34) AND volume spike
            if (trix[i-1] <= 0 and trix[i] > 0 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero AND downtrend (price < EMA34) AND volume spike
            elif (trix[i-1] >= 0 and trix[i] < 0 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero OR trend reversal (price < EMA34)
            if trix[i-1] >= 0 and trix[i] < 0:
                signals[i] = 0.0
                position = 0
            elif close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero OR trend reversal (price > EMA34)
            if trix[i-1] <= 0 and trix[i] > 0:
                signals[i] = 0.0
                position = 0
            elif close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals