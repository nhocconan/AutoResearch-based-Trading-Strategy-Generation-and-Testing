#!/usr/bin/env python3
"""
12h_TRIX_With_Volume_Spike_and_Chop_Regime
Hypothesis: Use TRIX (1-period ROC of EMA) for momentum, volume spike for conviction, and Choppiness Index for regime filter.
Long when TRIX crosses above zero + volume spike + chop > 61.8 (ranging market for mean reversion).
Short when TRIX crosses below zero + volume spike + chop > 61.8.
Exit when TRIX crosses back across zero or chop < 38.2 (trending regime).
Designed to work in both bull and bear markets by focusing on mean reversion in ranging conditions.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

name = "12h_TRIX_With_Volume_Spike_and_Chop_Regime"
timeframe = "12h"
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
    
    # Get daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA20 for trend context (not direct signal)
    close_daily = df_daily['close'].values
    ema20_daily = np.full_like(close_daily, np.nan)
    if len(close_daily) >= 20:
        multiplier = 2.0 / (20 + 1)
        ema20_daily[19] = np.mean(close_daily[:20])
        for i in range(20, len(close_daily)):
            ema20_daily[i] = multiplier * close_daily[i] + (1 - multiplier) * ema20_daily[i-1]
    ema20_daily_aligned = align_htf_to_ltf(prices, df_daily, ema20_daily)
    
    # Calculate TRIX (1-period rate of change of triple EMA)
    # EMA1
    ema1 = np.zeros(n)
    ema1[0] = close[0]
    alpha1 = 2.0 / (12 + 1)
    for i in range(1, n):
        ema1[i] = alpha1 * close[i] + (1 - alpha1) * ema1[i-1]
    
    # EMA2 of EMA1
    ema2 = np.zeros(n)
    ema2[0] = ema1[0]
    alpha2 = 2.0 / (12 + 1)
    for i in range(1, n):
        ema2[i] = alpha2 * ema1[i] + (1 - alpha2) * ema2[i-1]
    
    # EMA3 of EMA2
    ema3 = np.zeros(n)
    ema3[0] = ema2[0]
    alpha3 = 2.0 / (12 + 1)
    for i in range(1, n):
        ema3[i] = alpha3 * ema2[i] + (1 - alpha3) * ema3[i-1]
    
    # TRIX = 1-period ROC of EMA3
    trix = np.zeros(n)
    trix[0] = 0
    for i in range(1, n):
        if ema3[i-1] != 0:
            trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
        else:
            trix[i] = 0
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    volume_spike = volume > (vol_ma20 * 1.5)
    
    # Choppiness Index (14-period)
    chop = np.full(n, 50.0)  # default neutral
    atr14 = np.zeros(n)
    tr = np.zeros(n)
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(13, n):
        atr14[i] = np.mean(tr[i-13:i+1])
    
    for i in range(13, n):
        if atr14[i] > 0 and (high[i] - low[i]) > 0:
            sum_tr = np.sum(tr[i-13:i+1])
            max_high = np.max(high[i-13:i+1])
            min_low = np.min(low[i-13:i+1])
            if max_high > min_low:
                chop[i] = 100 * np.log10(sum_tr / (max_high - min_low)) / np.log10(14)
            else:
                chop[i] = 50.0
        else:
            chop[i] = 50.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(ema20_daily_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero + volume spike + chop > 61.8 (ranging)
            if trix[i] > 0 and trix[i-1] <= 0 and volume_spike[i] and chop[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + volume spike + chop > 61.8 (ranging)
            elif trix[i] < 0 and trix[i-1] >= 0 and volume_spike[i] and chop[i] > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TRIX crosses below zero OR chop < 38.2 (trending regime)
            if trix[i] < 0 and trix[i-1] >= 0 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX crosses above zero OR chop < 38.2 (trending regime)
            if trix[i] > 0 and trix[i-1] <= 0 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals