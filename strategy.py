#!/usr/bin/env python3
# 4h_TRIX_ZeroCross_1dTrend_VolumeSpike
# Hypothesis: TRIX zero-cross signals aligned with 1-day trend and volume spikes capture momentum with low trade frequency.
# TRIX (12-period) filters noise, zero-cross indicates momentum shift. 
# Requires alignment with 1-day EMA50 trend and volume >2x 20-period average to avoid false signals.
# Designed for <50 trades/year to minimize fee drag in BTC/ETH markets.
# Works in bull markets via long signals and bear markets via short signals following daily trend.

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
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1-day EMA50 for trend filter
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (ema_50_1d[i-1] * 49 + close_1d[i]) / 50
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate TRIX (12-period) on 4h close
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - then % change
    ema1 = np.full_like(close, np.nan)
    ema2 = np.full_like(close, np.nan)
    ema3 = np.full_like(close, np.nan)
    
    if len(close) >= 12:
        ema1[11] = np.mean(close[0:12])
        ema2[11] = np.mean(close[0:12])
        ema3[11] = np.mean(close[0:12])
        for i in range(12, len(close)):
            ema1[i] = (ema1[i-1] * 11 + close[i]) / 12
            ema2[i] = (ema2[i-1] * 11 + ema1[i]) / 12
            ema3[i] = (ema2[i-1] * 11 + ema2[i]) / 12
    
    trix = np.full_like(close, np.nan)
    if len(close) >= 13:  # Need at least one prior value for ROC
        # Calculate 1-period % change of triple EMA
        for i in range(12, len(close)):
            if not np.isnan(ema3[i]) and not np.isnan(ema3[i-1]) and ema3[i-1] != 0:
                trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Ensure volume MA and TRIX are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: TRIX crosses above zero AND uptrend (price > EMA50) AND volume spike
            if (trix[i-1] <= 0 and trix[i] > 0 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero AND downtrend (price < EMA50) AND volume spike
            elif (trix[i-1] >= 0 and trix[i] < 0 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero OR trend reversal (price < EMA50)
            if (trix[i-1] >= 0 and trix[i] < 0) or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero OR trend reversal (price > EMA50)
            if (trix[i-1] <= 0 and trix[i] > 0) or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals