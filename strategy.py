#!/usr/bin/env python3
"""
4h_1d_TRIX_VolumeSpike_Regime
Hypothesis: TRIX on 4h with volume spike confirmation and 1d Choppiness regime filter.
Long when TRIX crosses above zero with volume > 2x average and 1d chop > 61.8 (range).
Short when TRIX crosses below zero with volume > 2x average and 1d chop < 38.2 (trend).
Exit on opposite TRIX cross.
Designed for 4h to capture momentum in ranging markets and avoid whipsaws in strong trends.
Volume spike filters low-conviction moves. Regime filter adapts to market conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate TRIX on 4h (15-period EMA of 15-period EMA of 15-period EMA of close, then ROC)
    close = prices['close'].values
    # First EMA
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Second EMA
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Third EMA
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX = 100 * (ema3 - ema3_previous) / ema3_previous
    trix_raw = np.full_like(close, np.nan)
    trix_raw[15:] = 100 * (ema3[15:] - ema3[14:-1]) / ema3[14:-1]
    
    # Load 1d data for Choppiness index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Choppiness Index (14-period)
    atr_1d = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        atr_1d[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    # Smoothed ATR (14-period)
    atr_smoothed = pd.Series(atr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # Chop = 100 * log10(sum(atr14) / (max(high14) - min(low14))) / log10(14)
    sum_atr14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    chop = np.full_like(close_1d, np.nan)
    denominator = highest_high - lowest_low
    chop[13:] = 100 * np.log10(sum_atr14[13:] / denominator[13:]) / np.log10(14)
    
    # Align TRIX and Chop to 4h
    trix_aligned = align_htf_to_ltf(prices, prices, trix_raw)  # Already 4h, no alignment needed but safe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(trix_aligned[i-1])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 2x 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        # TRIX crossover signals
        trix_cross_up = trix_aligned[i-1] <= 0 and trix_aligned[i] > 0
        trix_cross_down = trix_aligned[i-1] >= 0 and trix_aligned[i] < 0
        
        if position == 0:
            # Long conditions: TRIX crosses up + volume + chop > 61.8 (range)
            if trix_cross_up and volume_ok and chop_aligned[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # Short conditions: TRIX crosses down + volume + chop < 38.2 (trend)
            elif trix_cross_down and volume_ok and chop_aligned[i] < 38.2:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TRIX crosses below zero
            if trix_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX crosses above zero
            if trix_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_TRIX_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0