#!/usr/bin/env python3
"""
4h_TRIX_Zero_Cross_With_Volume_Spike_and_1dTrend
Hypothesis: Go long when TRIX crosses above zero with volume spike and price above 1d EMA50; short when TRIX crosses below zero with volume spike and price below 1d EMA50. TRIX is a momentum oscillator that filters out insignificant price movements, effective in trending markets. Volume spike confirms institutional participation, and 1d EMA50 ensures alignment with long-term trend. Designed for low trade frequency to minimize fee drain while capturing high-probability momentum shifts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf  # Note: align_ltf_to_htf is not used; using align_htf_to_ltf as per standard

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate TRIX (1-period ROC of triple EMA)
    # TRIX = 100 * (EMA3 - EMA3_prev) / EMA3_prev
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0  # First value has no previous
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Need EMA50 and TRIX warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema50_val = ema50_1d_aligned[i]
        vol_spike = volume_spike[i]
        trix_now = trix[i]
        trix_prev = trix[i-1] if i > 0 else 0
        
        if position == 0:
            # Long: TRIX crosses above zero with volume spike and above 1d EMA50
            if trix_now > 0 and trix_prev <= 0 and vol_spike and price > ema50_val:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume spike and below 1d EMA50
            elif trix_now < 0 and trix_prev >= 0 and vol_spike and price < ema50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: TRIX crosses below zero or price below 1d EMA50
            if trix_now < 0 and trix_prev >= 0 or price < ema50_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: TRIX crosses above zero or price above 1d EMA50
            if trix_now > 0 and trix_prev <= 0 or price > ema50_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TRIX_Zero_Cross_With_Volume_Spike_and_1dTrend"
timeframe = "4h"
leverage = 1.0