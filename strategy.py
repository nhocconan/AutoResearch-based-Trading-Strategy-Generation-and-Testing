#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_TRIX_ZeroCross_Volume_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === TRIX (15-period EMA of EMA of EMA of close) on 1d ===
    close_1d = df_1d['close'].values
    # EMA1
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA2
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA3
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX = (EMA3 - previous EMA3) / previous EMA3 * 100
    trix_raw = np.zeros_like(ema3)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / np.where(np.abs(ema3[:-1]) > 1e-10, np.abs(ema3[:-1]), 1e-10) * 100
    
    # Align TRIX to 12h
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix_raw)
    
    # === Volume Confirmation (12h) ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        trix_val = trix_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(trix_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero with volume confirmation
            if trix_val > 0 and trix_aligned[i-1] <= 0 and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume confirmation
            elif trix_val < 0 and trix_aligned[i-1] >= 0 and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero
            if trix_val < 0 and trix_aligned[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero
            if trix_val > 0 and trix_aligned[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals