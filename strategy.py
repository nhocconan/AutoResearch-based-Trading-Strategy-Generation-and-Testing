#!/usr/bin/env python3
# 4h_1d_Trix_VolumeSpike_TrendFilter
# Hypothesis: TRIX momentum on 1d timeframe combined with volume spike and 1d EMA trend filter to capture sustained moves in both bull and bear markets.
# Uses 4h for entry timing with strict conditions to limit trades (target: 20-40/year) and avoid fee drag.
# TRIX filters out noise, volume spike confirms institutional interest, EMA ensures trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Trix_VolumeSpike_TrendFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for TRIX, EMA, and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === Calculate 1d EMA12 of close ===
    ema12_1d = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # === Calculate 1d EMA12 of EMA12 (for TRIX) ===
    ema12_of_ema12_1d = pd.Series(ema12_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # === Calculate 1d EMA12 of EMA12 of EMA12 (for TRIX) ===
    ema12_of_ema12_of_ema12_1d = pd.Series(ema12_of_ema12_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # === Calculate TRIX: 100 * (EMA12_of_ema12_of_ema12 - prev) / prev ===
    trix_raw = 100 * (ema12_of_ema12_of_ema12_1d - np.roll(ema12_of_ema12_of_ema12_1d, 1)) / np.roll(ema12_of_ema12_of_ema12_1d, 1)
    # Handle first value (no previous)
    trix_raw[0] = 0
    
    # === Calculate 1d EMA9 of TRIX (signal line) ===
    trix_signal = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # === Calculate 1d EMA34 for trend filter ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === Calculate 1d volume EMA20 for spike detection ===
    vol_ema20_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === Align all 1d indicators to 4h ===
    trix_signal_aligned = align_htf_to_ltf(prices, df_1d, trix_signal)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema20_1d)
    
    # === 4h: Volume ratio (current vs 20-period EMA) ===
    volume = prices['volume'].values
    vol_ema20_4h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ema20_4h > 0, vol_ema20_4h, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA and TRIX warmup
        # Get values
        close_val = prices['close'].iloc[i]
        trix_val = trix_signal_aligned[i]
        ema34_1d_val = ema34_1d_aligned[i]
        vol_ema20_1d_val = vol_ema20_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(trix_val) or np.isnan(ema34_1d_val) or 
            np.isnan(vol_ema20_1d_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above signal line with volume spike and above EMA34
            # Use TRIX > 0 and rising as entry condition to avoid whipsaws
            if (trix_val > 0 and trix_val > trix_signal[i-1] if i > 0 else False and 
                vol_ratio_val > 2.0 and close_val > ema34_1d_val):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal line with volume spike and below EMA34
            elif (trix_val < 0 and trix_val < trix_signal[i-1] if i > 0 else False and 
                  vol_ratio_val > 2.0 and close_val < ema34_1d_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TRIX turns negative or volume dries up
            if trix_val < 0 or vol_ratio_val < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX turns positive or volume dries up
            if trix_val > 0 or vol_ratio_val < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals