#!/usr/bin/env python3
# 6h_12h_cci_reversal_v1
# Hypothesis: 6-hour reversals at weekly CCI extremes (>150 or <-150) with momentum divergence.
# Weekly CCI >150 indicates overbought (fading long, taking short); <-150 oversold (fading short, taking long).
# Works in ranging markets (mean reversion) and trending markets (pullbacks in trend).
# Volume filter confirms institutional interest at extremes. Target: 20-50 trades per year per symbol (~80-200 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_cci_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly CCI(20)
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    tp_mean = typical_price.rolling(window=20, min_periods=20).mean()
    tp_mad = typical_price.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (typical_price - tp_mean) / (0.015 * tp_mad)
    cci_values = cci.values
    
    # Align weekly CCI to 6h timeframe (no extra delay - CCI is contemporaneous)
    cci_aligned = align_htf_to_ltf(prices, df_1w, cci_values)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(cci_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI returns below 100 or volume drops
            if cci_aligned[i] < 100 or volume[i] < vol_ma_20[i] * 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI returns above -100 or volume drops
            if cci_aligned[i] > -100 or volume[i] < vol_ma_20[i] * 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: CCI < -150 (oversold) with volume confirmation
            if cci_aligned[i] < -150 and volume[i] > vol_ma_20[i] * 1.5:
                position = 1
                signals[i] = 0.25
            # Enter short: CCI > 150 (overbought) with volume confirmation
            elif cci_aligned[i] > 150 and volume[i] > vol_ma_20[i] * 1.5:
                position = -1
                signals[i] = -0.25
    
    return signals