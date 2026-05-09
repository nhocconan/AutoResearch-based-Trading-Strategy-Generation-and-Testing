#!/usr/bin/env python3
# 2025-06-22 | 4h_Trix_Signal_Line_Cross_12hTrend_VolumeSpike
# Hypothesis: TRIX crossing its signal line (EMA of TRIX) with 12h EMA50 trend filter and volume spike confirmation.
# TRIX is a momentum oscillator that filters out insignificant price movements; its signal line cross indicates momentum shifts.
# Combining with 12h trend ensures trades align with higher timeframe momentum, reducing whipsaw.
# Volume spike (>2x 24-period average) confirms breakout strength. Designed for low trade frequency (20-50/year).

name = "4h_Trix_Signal_Line_Cross_12hTrend_VolumeSpike"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[0:50])
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = (ema_50_12h[i-1] * 49 + close_12h[i]) / 50
    
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate TRIX: triple EMA of ROC, then signal line (EMA of TRIX)
    # ROC period = 12
    roc = np.full_like(close, np.nan)
    if len(close) >= 13:
        roc[12:] = (close[12:] - close[:-12]) / close[:-12] * 100
    
    # EMA1 of ROC
    ema1 = np.full_like(roc, np.nan)
    if len(roc) >= 12:
        ema1[11] = np.mean(roc[0:12])
        for i in range(12, len(roc)):
            if not np.isnan(roc[i]):
                ema1[i] = (roc[i] * 2 + ema1[i-1] * (12-1)) / (12+1)
    
    # EMA2 of EMA1
    ema2 = np.full_like(ema1, np.nan)
    if len(ema1) >= 12:
        ema2[11] = np.mean(ema1[0:12])
        for i in range(12, len(ema1)):
            if not np.isnan(ema1[i]):
                ema2[i] = (ema1[i] * 2 + ema2[i-1] * (12-1)) / (12+1)
    
    # EMA3 of EMA2 = TRIX
    trix = np.full_like(ema2, np.nan)
    if len(ema2) >= 12:
        trix[11] = np.mean(ema2[0:12])
        for i in range(12, len(ema2)):
            if not np.isnan(ema2[i]):
                trix[i] = (ema2[i] * 2 + ema2[i-1] * (12-1)) / (12+1)
    
    # Signal line = EMA of TRIX (period=9)
    signal_line = np.full_like(trix, np.nan)
    if len(trix) >= 9:
        signal_line[8] = np.mean(trix[0:9])
        for i in range(9, len(trix)):
            if not np.isnan(trix[i]):
                signal_line[i] = (trix[i] * 2 + signal_line[i-1] * (9-1)) / (9+1)
    
    # Volume spike filter: current volume / 24-period average volume (24*4h = 4 days)
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 24:
        vol_ma[23] = np.mean(volume[0:24])
        for i in range(24, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 23 + volume[i]) / 24
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(24, 12+12+12+9)  # Ensure TRIX, signal line, volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(signal_line[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: TRIX crosses above signal line AND uptrend (price > EMA50) AND volume spike
            if (trix[i] > signal_line[i] and trix[i-1] <= signal_line[i-1] and 
                close[i] > ema_50_12h_aligned[i] and volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: TRIX crosses below signal line AND downtrend (price < EMA50) AND volume spike
            elif (trix[i] < signal_line[i] and trix[i-1] >= signal_line[i-1] and 
                  close[i] < ema_50_12h_aligned[i] and volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit long: TRIX crosses below signal line OR trend reversal (price < EMA50)
            if (trix[i] < signal_line[i] and trix[i-1] >= signal_line[i-1]) or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above signal line OR trend reversal (price > EMA50)
            if (trix[i] > signal_line[i] and trix[i-1] <= signal_line[i-1]) or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals