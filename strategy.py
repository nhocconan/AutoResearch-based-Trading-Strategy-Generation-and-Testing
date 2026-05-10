#!/usr/bin/env python3
# 4h_TRIX_Zero_Cross_With_Volume_and_CHOP
# Hypothesis: TRIX (triple exponential moving average) crossing zero indicates momentum shifts.
# Combines with volume confirmation (>1.5x 20-period volume average) and Choppiness Index
# regime filter (CHOP > 61.8 for range, < 38.2 for trend) to avoid false signals.
# Works in bull markets (rides momentum) and bear markets (catches reversals) by
# only taking signals aligned with higher timeframe trend (1d EMA50).
# Target: 20-50 trades/year on 4h timeframe to minimize fee drag.

name = "4h_TRIX_Zero_Cross_With_Volume_and_CHOP"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate TRIX on 4h data (15-period)
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = np.diff(ema3, prepend=ema3[0]) / ema3 * 100  # percentage change
    
    # Calculate volume moving average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (14-period) on 4h data
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14 * 14 / (highest_high14 - lowest_low14)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need TRIX (45), EMA50 (50), volume MA (20), ATR14 (14), HH/LL (14)
    start_idx = max(50, 45, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(trix[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher timeframe trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Chop regime: only trade in trending markets (CHOP < 38.2)
        trending = chop[i] < 38.2
        
        if position == 0:
            # Long entry: TRIX crosses above zero + uptrend + volume + trending
            if trix[i] > 0 and trix[i-1] <= 0 and uptrend and volume_confirm and trending:
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX crosses below zero + downtrend + volume + trending
            elif trix[i] < 0 and trix[i-1] >= 0 and downtrend and volume_confirm and trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero or trend breaks
            if trix[i] < 0 and trix[i-1] >= 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero or trend breaks
            if trix[i] > 0 and trix[i-1] <= 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals