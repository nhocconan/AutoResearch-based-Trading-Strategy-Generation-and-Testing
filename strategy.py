#!/usr/bin/env python3
"""
6h_1d_KAMA_Trend_Volume_Momentum_v3
Hypothesis: KAMA(10) trend filter from 1d timeframe combined with 6h momentum (ROC > 0) and volume confirmation (volume > 20-period average). 
Enters long when: 6h close > KAMA(10) from prior 1d, ROC(10) > 0, volume > 20-period average.
Enters short when: 6h close < KAMA(10) from prior 1d, ROC(10) < 0, volume > 20-period average.
Exits when trend reverses (price crosses KAMA) or volume drops below average.
Uses KAMA's adaptive smoothing to reduce whipsaw in choppy markets while capturing trends.
Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
"""

name = "6h_1d_KAMA_Trend_Volume_Momentum_v3"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get 1d data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- 1d KAMA(10) Trend Filter ---
    close_1d = df_1d['close'].values
    # Calculate Efficiency Ratio (ER) and Smoothing Constant (SC)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    er[1:] = change[1:] / (np.abs(volatility).rolling(window=10, min_periods=1).sum().values + 1e-10)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    kama_1d = np.zeros_like(close_1d)
    kama_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama_1d[i] = kama_1d[i-1] + sc[i] * (close_1d[i] - kama_1d[i-1])
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # --- 6h Momentum: ROC(10) ---
    roc_10 = np.zeros_like(close_6h)
    roc_10[10:] = (close_6h[10:] - close_6h[:-10]) / close_6h[:-10]
    
    # --- Volume Confirmation: 6h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = max(20, 10)  # for volume MA and ROC
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(roc_10[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_ok = volume_6h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries with momentum and volume confirmation
            if close_6h[i] > kama_1d_aligned[i] and roc_10[i] > 0 and vol_ok:
                # Long: price above KAMA, positive momentum, volume confirmation
                signals[i] = 0.25
                position = 1
            elif close_6h[i] < kama_1d_aligned[i] and roc_10[i] < 0 and vol_ok:
                # Short: price below KAMA, negative momentum, volume confirmation
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal or volume drop
            if position == 1:
                # Exit long: price crosses below KAMA or momentum turns negative
                if close_6h[i] <= kama_1d_aligned[i] or roc_10[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above KAMA or momentum turns positive
                if close_6h[i] >= kama_1d_aligned[i] or roc_10[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals