#!/usr/bin/env python3
# 4H_Trix_Signal_With_Volume_and_Chop_Filter
# Hypothesis: Uses TRIX (12-period) as a momentum oscillator combined with volume confirmation and
# Choppiness Index regime filter to avoid whipsaws in sideways markets. Designed for 4h timeframe
# to capture momentum bursts in both bull and bear markets while filtering out low-quality signals.
# Targets 20-40 trades per year by requiring TRIX crossover, volume > 1.5x average, and CHOP < 61.8 (trending regime).
# Uses discrete position sizing (0.25) to minimize churn and improve generalization.

name = "4H_Trix_Signal_With_Volume_and_Chop_Filter"
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
    
    # Calculate TRIX (12-period) on close
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) then percent change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = pd.Series(ema3).pct_change(1).values * 100  # percent change
    
    # Calculate Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR, 14) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # first period has no prior close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low + 1e-9)) / np.log10(14)
    
    # Volume filter: volume > 1.5x 34-period average
    vol_ma = pd.Series(volume).rolling(window=34, min_periods=34).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(36, 34, 14)  # Warmup for TRIX (36), volume MA (34), CHOP (14)
    
    for i in range(start_idx, n):
        if np.isnan(trix[i]) or np.isnan(chop[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # TRIX signal: crossover above/below zero line
        trix_cross_up = trix[i] > 0 and trix[i-1] <= 0
        trix_cross_down = trix[i] < 0 and trix[i-1] >= 0
        
        if position == 0:
            # Long entry: TRIX crosses up + volume spike + trending market (CHOP < 61.8)
            if (trix_cross_up and 
                volume[i] > vol_threshold[i] and 
                chop[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX crosses down + volume spike + trending market (CHOP < 61.8)
            elif (trix_cross_down and 
                  volume[i] > vol_threshold[i] and 
                  chop[i] < 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses down or volume drops below average or chop becomes too high (ranging)
            if (trix_cross_down or 
                volume[i] < vol_ma[i] or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses up or volume drops below average or chop becomes too high (ranging)
            if (trix_cross_up or 
                volume[i] < vol_ma[i] or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals