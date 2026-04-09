#!/usr/bin/env python3
# 6h_1d_cci_reversal_v1
# Hypothesis: 6-hour reversals at daily CCI extremes with volume confirmation.
# In bull markets, buy dips when CCI < -100; in bear markets, sell rallies when CCI > +100.
# Uses CCI(20) on daily timeframe to identify overextended moves and mean reversion opportunities.
# Volume filter ensures institutional participation, reducing false signals.
# Designed for low trade frequency (15-35 trades/year) to minimize fee drag in 6h timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_cci_reversal_v1"
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
    
    # Load 1d data ONCE before loop for CCI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate CCI(20) on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical Price
    tp_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # 20-period SMA of TP
    sma_tp = np.full(len(tp_1d), np.nan)
    for i in range(len(tp_1d)):
        if i >= 19:
            sma_tp[i] = np.mean(tp_1d[i-19:i+1])
    
    # Mean Deviation
    md = np.full(len(tp_1d), np.nan)
    for i in range(len(tp_1d)):
        if i >= 19:
            dev = np.abs(tp_1d[i-19:i+1] - sma_tp[i])
            md[i] = np.mean(dev)
    
    # CCI calculation
    cci_1d = np.full(len(tp_1d), np.nan)
    for i in range(len(tp_1d)):
        if i >= 19 and md[i] > 0:
            cci_1d[i] = (tp_1d[i] - sma_tp[i]) / (0.015 * md[i])
    
    # Align CCI to 6h timeframe
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    
    # Volume confirmation - 20 period average on 6h
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
    
    for i in range(20, n):  # Start after CCI warmup
        # Skip if any required data is invalid
        if np.isnan(cci_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI returns above -100 (mean reversion complete)
            if cci_aligned[i] > -100:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI returns below +100 (mean reversion complete)
            if cci_aligned[i] < 100:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: CCI below -100 (oversold) with volume confirmation
            if cci_aligned[i] < -100 and volume[i] > vol_ma_20[i] * 1.5:
                position = 1
                signals[i] = 0.25
            # Enter short: CCI above +100 (overbought) with volume confirmation
            elif cci_aligned[i] > 100 and volume[i] > vol_ma_20[i] * 1.5:
                position = -1
                signals[i] = -0.25
    
    return signals