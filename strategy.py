#!/usr/bin/env python3
"""
12h_Trix_VolumeSpike_TrendFilter
Hypothesis: TRIX momentum with volume spikes identifies strong breakouts in trending markets. Using 1-day trend filter (EMA34) ensures alignment with higher timeframe momentum, reducing false signals. Works in bull/bear by only taking long when price > EMA34 and short when price < EMA34. Volume confirmation filters low-conviction moves. Target 20-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1-day data once for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = np.zeros_like(close_1d)
    ema34_1d[0] = close_1d[0]
    alpha = 2.0 / (34 + 1)
    for i in range(1, len(close_1d)):
        ema34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema34_1d[i-1]
    
    # Align daily EMA34 to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate TRIX (15-period EMA of EMA of EMA of ROC)
    close = prices['close'].values
    # Step 1: ROC 1-period
    roc = np.zeros(n)
    roc[0] = 0
    for i in range(1, n):
        roc[i] = (close[i] - close[i-1]) / close[i-1] * 100
    # Step 2: Triple EMA
    ema1 = np.zeros(n)
    ema2 = np.zeros(n)
    ema3 = np.zeros(n)
    ema1[0] = roc[0]
    ema2[0] = roc[0]
    ema3[0] = roc[0]
    alpha_trix = 2.0 / (15 + 1)
    for i in range(1, n):
        ema1[i] = alpha_trix * roc[i] + (1 - alpha_trix) * ema1[i-1]
        ema2[i] = alpha_trix * ema1[i] + (1 - alpha_trix) * ema2[i-1]
        ema3[i] = alpha_trix * ema2[i] + (1 - alpha_trix) * ema3[i-1]
    trix = ema3  # TRIX is the final smoothed EMA
    
    # Volume filter: volume > 1.8x 20-period average
    volume = prices['volume'].values
    vol_avg = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            vol_avg[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_avg[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(trix[i]) or np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        trix_val = trix[i]
        ema34 = ema34_1d_aligned[i]
        vol_confirm = volume_filter[i]
        
        # Exit conditions
        if position == 1:
            if trix_val < 0 or price < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
            continue
        elif position == -1:
            if trix_val > 0 or price > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
            continue
        
        # Entry conditions
        if position == 0:
            # Long: TRIX positive with volume confirmation in uptrend (price > EMA34)
            if trix_val > 0 and vol_confirm and price > ema34:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: TRIX negative with volume confirmation in downtrend (price < EMA34)
            elif trix_val < 0 and vol_confirm and price < ema34:
                signals[i] = -0.25
                position = -1
                entry_price = price
    
    return signals

name = "12h_Trix_VolumeSpike_TrendFilter"
timeframe = "12h"
leverage = 1.0