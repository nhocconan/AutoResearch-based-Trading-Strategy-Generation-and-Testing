#!/usr/bin/env python3
"""
4h_12h_trix_volume_regime_v1
Hypothesis: TRIX (Triple Exponential Average) with volume confirmation and chop regime filter on 4h timeframe.
TRIX filters out insignificant cycles and highlights sustained momentum. Works in bull/bear markets by
using volume to confirm breakouts and chop regime to avoid whipsaws in ranging conditions.
Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
"""

name = "4h_12h_trix_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for TRIX and chop regime
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate TRIX (15-period triple EMA)
    ema1 = pd.Series(close_12h).ewm(span=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, adjust=False).mean()
    trix = 100 * (ema3.pct_change())
    trix_signal = trix.ewm(span=9, adjust=False).mean()
    trix_hist = trix - trix_signal
    trix_values = trix_hist.values
    
    # Calculate Chopiness Index (14-period) for regime filter
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    max_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop_values = chop.values
    
    # Align TRIX histogram and Chop to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix_values)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop_values)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: TRIX histogram crosses above zero with volume and in trending regime (chop < 61.8)
        if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and vol_confirm[i] and 
            chop_aligned[i] < 61.8 and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: TRIX histogram crosses below zero with volume and in trending regime (chop < 61.8)
        elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and vol_confirm[i] and 
              chop_aligned[i] < 61.8 and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: TRIX histogram crosses zero in opposite direction or chop > 61.8 (ranging market)
        elif position == 1 and (trix_aligned[i] < 0 or chop_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (trix_aligned[i] > 0 or chop_aligned[i] > 61.8):
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