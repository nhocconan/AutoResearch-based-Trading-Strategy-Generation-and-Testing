#!/usr/bin/env python3
# 4h_Trix_Signal_Line_Cross_With_Volume_Filter
# Hypothesis: TRIX (triple exponential average) signal line cross with volume confirmation.
# TRIX filters out insignificant price movements and shows smoothed momentum.
# Bullish when TRIX crosses above its signal line (EMA of TRIX), bearish when crosses below.
# Volume spike (>1.5x average) confirms institutional participation.
# Works in both bull/bear markets by following momentum with volume confirmation.

name = "4h_Trix_Signal_Line_Cross_With_Volume_Filter"
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
    
    # Calculate TRIX: triple EMA of close, then percent change
    # TRIX period: 12, Signal line period: 9
    def ema(array, period):
        result = np.full_like(array, np.nan)
        if len(array) < period:
            return result
        multiplier = 2 / (period + 1)
        result[period-1] = np.mean(array[0:period])
        for i in range(period, len(array)):
            result[i] = (array[i] - result[i-1]) * multiplier + result[i-1]
        return result
    
    # First EMA
    ema1 = ema(close, 12)
    # Second EMA of EMA1
    ema2 = ema(ema1, 12)
    # Third EMA of EMA2
    ema3 = ema(ema2, 12)
    
    # TRIX = percentage change of ema3
    trix = np.full_like(close, np.nan)
    valid = ~np.isnan(ema3)
    trix[valid] = (ema3[valid] - np.roll(ema3[valid], 1)) / np.roll(ema3[valid], 1) * 100
    # Handle first value
    if not np.isnan(ema3[0]):
        trix[0] = 0
    
    # Signal line = EMA of TRIX
    signal_line = ema(trix, 9)
    
    # Volume spike filter: current volume / 20-period average
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
    
    start_idx = max(30, 20)  # Ensure TRIX, signal line, and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(signal_line[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: TRIX crosses above signal line AND volume spike
            if trix[i] > signal_line[i] and trix[i-1] <= signal_line[i-1] and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below signal line AND volume spike
            elif trix[i] < signal_line[i] and trix[i-1] >= signal_line[i-1] and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below signal line
            if trix[i] < signal_line[i] and trix[i-1] >= signal_line[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above signal line
            if trix[i] > signal_line[i] and trix[i-1] <= signal_line[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals