#!/usr/bin/env python3
# 4h_1d_trix_volume_reversal_v1
# Strategy: 4h TRIX momentum reversal with 1d volume spike and 1d choppiness regime filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: TRIX captures momentum reversals in both bull and bear markets.
# Volume spike confirms institutional participation, while choppiness filter avoids
# false signals in strong trends. Designed for low trade frequency (~20-30/year)
# to minimize fee drag and improve generalization across market regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_trix_volume_reversal_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d TRIX (15-period EMA of EMA of EMA of close)
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = np.where(ema3[:-1] != 0, (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100, 0)
    trix = np.concatenate([np.array([0.0]), trix])  # align length
    
    # Calculate 1d choppiness index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        hl = high_1d[i] - low_1d[i]
        hc = abs(high_1d[i] - close_1d[i-1])
        lc = abs(low_1d[i] - close_1d[i-1])
        atr_1d[i] = max(hl, hc, lc)
    atr_1d[0] = atr_1d[1] if len(atr_1d) > 1 else 0
    
    sum_tr14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = highest_high_14 - lowest_low_14
    chop = np.where(range_14 != 0, -100 * np.log10(sum_tr14 / range_14) / np.log10(14), 50)
    chop = np.nan_to_num(chop, nan=50.0)
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 4h volume > 2.0x 1d average volume (scaled)
        # Scale 1d volume to 4h: approx 1/6 of 1d volume per 4h bar
        vol_scaled = vol_avg_20_1d_aligned[i] / 6.0
        vol_confirm = volume[i] > 2.0 * vol_scaled
        
        # TRIX momentum signals
        trix_up = trix_aligned[i] > 0 and trix_aligned[i-1] <= 0
        trix_down = trix_aligned[i] < 0 and trix_aligned[i-1] >= 0
        
        # Choppiness regime filter: only trade in choppy markets (CHOP > 61.8)
        choppy = chop_aligned[i] > 61.8
        
        # Entry conditions
        # Long: TRIX crosses above zero AND choppy market AND volume confirmation
        if trix_up and choppy and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: TRIX crosses below zero AND choppy market AND volume confirmation
        elif trix_down and choppy and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: TRIX crosses back through zero in opposite direction
        elif position == 1 and trix_aligned[i] < 0:
            position = 0
            signals[i] = 0.0
        elif position == -1 and trix_aligned[i] > 0:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals