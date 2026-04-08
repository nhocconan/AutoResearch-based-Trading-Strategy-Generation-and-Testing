#!/usr/bin/env python3
"""
4h_1d_trix_volume_regime_v1
Hypothesis: Use TRIX momentum on 1d for trend bias, TRIX crossovers on 4h for entry timing, volume confirmation, and choppiness regime filter to avoid whipsaws. Long when 4h TRIX crosses above zero with volume and 1d TRIX > 0 and choppiness < 61.8. Short when 4h TRIX crosses below zero with volume and 1d TRIX < 0 and choppiness < 61.8. Designed to capture momentum in trending markets while avoiding range-bound chop.
Target: 20-50 trades/year per symbol (80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_trix_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for TRIX and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d TRIX (15-period EMA of EMA of EMA of close, then ROC)
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix_1d = 100 * (pd.Series(ema3).pct_change().values)  # ROC of triple EMA
    
    # Calculate 1d choppiness index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = np.zeros(len(close_1d))
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(sum_tr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    
    # Align 1d TRIX and choppiness to 4h
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 4h TRIX for entry signals
    ema1_4h = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2_4h = pd.Series(ema1_4h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3_4h = pd.Series(ema2_4h).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix_4h = 100 * (pd.Series(ema3_4h).pct_change().values)
    
    # Volume confirmation: volume > 1.3x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.3
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trix_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or np.isnan(trix_4h[i]) or
            np.isnan(trix_4h[i-1]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: 4h TRIX crosses below zero or choppiness too high
            if trix_4h[i] < 0 and trix_4h[i-1] >= 0 or chop_1d_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: 4h TRIX crosses above zero or choppiness too high
            if trix_4h[i] > 0 and trix_4h[i-1] <= 0 or chop_1d_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: 4h TRIX crosses above zero with volume, 1d TRIX bullish, low chop
            if trix_4h[i] > 0 and trix_4h[i-1] <= 0 and vol_confirm[i] and trix_1d_aligned[i] > 0 and chop_1d_aligned[i] < 61.8:
                position = 1
                signals[i] = 0.25
            # Short entry: 4h TRIX crosses below zero with volume, 1d TRIX bearish, low chop
            elif trix_4h[i] < 0 and trix_4h[i-1] >= 0 and vol_confirm[i] and trix_1d_aligned[i] < 0 and chop_1d_aligned[i] < 61.8:
                position = -1
                signals[i] = -0.25
    
    return signals