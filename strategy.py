#!/usr/bin/env python3
"""
6h_1d_trix_volume_momentum
Hypothesis: TRIX (15-period) captures smoothed momentum on 6h timeframe, 
filtered by 1d volume spike (>2x 20-day average) to avoid false signals.
TRIX > 0 with volume spike = long; TRIX < 0 with volume spike = short.
Uses volume confirmation to enter only during high conviction moves.
Works in bull/bear by requiring volume confirmation, avoiding chop.
Target: 20-40 trades/year (80-160 total over 4 years).
"""

name = "6h_1d_trix_volume_momentum"
timeframe = "6h"
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
    
    # Get daily data for TRIX and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # TRIX: Triple EMA of 15-period ROC
    # ROC(15) = (close / close.shift(15) - 1) * 100
    roc = np.zeros_like(close_1d)
    roc[15:] = (close_1d[15:] / close_1d[:-15] - 1) * 100
    
    # Triple EMA of ROC
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = ema3  # Already in percentage
    
    # Volume spike: >2x 20-day average
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma * 2.0)
    
    # Align TRIX and volume spike to 6h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(trix_aligned[i]) or np.isnan(vol_spike_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Long entry: positive TRIX with volume spike
        if trix_aligned[i] > 0 and vol_spike_aligned[i] > 0.5 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: negative TRIX with volume spike
        elif trix_aligned[i] < 0 and vol_spike_aligned[i] > 0.5 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: TRIX crosses zero or volume spike ends
        elif position == 1 and (trix_aligned[i] <= 0 or vol_spike_aligned[i] <= 0.5):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (trix_aligned[i] >= 0 or vol_spike_aligned[i] <= 0.5):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals