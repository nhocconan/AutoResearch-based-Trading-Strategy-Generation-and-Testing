#!/usr/bin/env python3
# 4h_TRIX_Zero_Cross_Volume_Spike_Trend_Filter
# Hypothesis: TRIX (triple smoothed EMA) zero cross with volume spike and trend filter.
# TRIX filters noise and captures momentum shifts. Zero cross indicates trend change.
# Volume spike confirms momentum strength. Trend filter (EMA50) avoids counter-trend trades.
# Works in bull/bear: trend filter ensures alignment with higher timeframe trend, volume confirms breakout strength.
# Target: 20-50 trades/year to minimize fee drag.

name = "4h_TRIX_Zero_Cross_Volume_Spike_Trend_Filter"
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
    
    # Get 1h data for TRIX calculation (more responsive than 4h)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:
        return np.zeros(n)
    
    close_1h = df_1h['close'].values
    
    # Calculate TRIX: triple EMA of percentage change
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12)
    ema1 = np.full_like(close_1h, np.nan)
    ema2 = np.full_like(close_1h, np.nan)
    ema3 = np.full_like(close_1h, np.nan)
    
    if len(close_1h) >= 12:
        ema1[11] = np.mean(close_1h[0:12])
        for i in range(12, len(close_1h)):
            ema1[i] = (close_1h[i] * 2 + ema1[i-1] * 11) / 12
        
        ema2[11] = np.mean(ema1[0:12])
        for i in range(12, len(close_1h)):
            ema2[i] = (ema1[i] * 2 + ema2[i-1] * 11) / 12
        
        ema3[11] = np.mean(ema2[0:12])
        for i in range(12, len(close_1h)):
            ema3[i] = (ema2[i] * 2 + ema3[i-1] * 11) / 12
        
        # Calculate percentage change of triple EMA
        pct_change = np.full_like(ema3, np.nan)
        valid = ~np.isnan(ema3)
        pct_change[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
        
        # Smooth the percentage change (another EMA)
        trix = np.full_like(pct_change, np.nan)
        if len(pct_change) >= 12:
            trix[11] = np.mean(pct_change[0:12])
            for i in range(12, len(pct_change)):
                trix[i] = (pct_change[i] * 2 + trix[i-1] * 11) / 12
    else:
        trix = np.full_like(close_1h, np.nan)
    
    # Align TRIX to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1h, trix)
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 49) / 50
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (volume[i] * 2 + vol_ma[i-1] * 19) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: TRIX crosses above zero AND uptrend (price > EMA50) AND volume spike
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero AND downtrend (price < EMA50) AND volume spike
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero OR trend reversal (price < EMA50)
            if trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero OR trend reversal (price > EMA50)
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals