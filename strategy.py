#!/usr/bin/env python3
# 12h_TRIX_VolumeSpike_1wTrend
# Hypothesis: TRIX momentum crossing zero with volume spike and weekly trend filter (price > weekly EMA20).
# TRIX filters noise and captures momentum shifts. Weekly trend ensures trades align with higher timeframe direction.
# Volume spike confirms institutional participation. Designed for 12-37 trades/year on 12h timeframe.

name = "12h_TRIX_VolumeSpike_1wTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate weekly EMA(20)
    ema_20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[0:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (close_1w[i] * 2 + ema_20_1w[i-1] * 18) / 20
    
    # Align weekly EMA to 12h timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate TRIX(12) on 12h closes: triple EMA then percent change
    # First EMA(12)
    ema1 = np.full(n, np.nan)
    if n >= 12:
        ema1[11] = np.mean(close[0:12])
        for i in range(12, n):
            ema1[i] = (close[i] * 2 + ema1[i-1] * 10) / 12
    # Second EMA(12)
    ema2 = np.full(n, np.nan)
    if n >= 24:
        ema2[23] = np.mean(ema1[12:24]) if not np.any(np.isnan(ema1[12:24])) else np.nan
        for i in range(24, n):
            if not np.isnan(ema1[i]) and not np.isnan(ema2[i-1]):
                ema2[i] = (ema1[i] * 2 + ema2[i-1] * 10) / 12
    # Third EMA(12)
    ema3 = np.full(n, np.nan)
    if n >= 36:
        ema3[35] = np.mean(ema2[24:36]) if not np.any(np.isnan(ema2[24:36])) else np.nan
        for i in range(36, n):
            if not np.isnan(ema2[i]) and not np.isnan(ema3[i-1]):
                ema3[i] = (ema2[i] * 2 + ema3[i-1] * 10) / 12
    # TRIX = 100 * (ema3[i] - ema3[i-1]) / ema3[i-1]
    trix = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(ema3[i]) and not np.isnan(ema3[i-1]) and ema3[i-1] != 0:
            trix[i] = 100 * (ema3[i] - ema3[i-1]) / ema3[i-1]
    
    # Volume ratio: current volume / 20-period average
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, n):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full(n, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(36, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(trix[i]) or np.isnan(trix[i-1]) or np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: TRIX crosses above zero AND volume spike AND weekly uptrend
            if trix[i-1] <= 0 and trix[i] > 0 and volume_ratio[i] > 2.0 and close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero AND volume spike AND weekly downtrend
            elif trix[i-1] >= 0 and trix[i] < 0 and volume_ratio[i] > 2.0 and close[i] < ema_20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero or weekly trend turns down
            if trix[i] < 0 or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero or weekly trend turns up
            if trix[i] > 0 or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals