#!/usr/bin/env python3
# 4h_GoldenCross_MA50_MA200_VolumeFilter
# Hypothesis: Golden Cross (MA50 crossing above MA200) and Death Cross (MA50 crossing below MA200)
# capture major trend changes in BTC/ETH. Volume filter ensures institutional participation.
# Works in both bull and bear markets by following the primary trend. Target: 15-30 trades/year.

name = "4h_GoldenCross_MA50_MA200_VolumeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data (same timeframe) for MA calculations
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate MA50 and MA200
    ma50 = pd.Series(close_4h).rolling(window=50, min_periods=50).mean().values
    ma200 = pd.Series(close_4h).rolling(window=200, min_periods=200).mean().values
    
    # Align to LTF
    ma50_aligned = align_htf_to_ltf(prices, df_4h, ma50)
    ma200_aligned = align_htf_to_ltf(prices, df_4h, ma200)
    
    # Volume filter: volume > 1.5x 50-period average
    vol_ma50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (vol_ma50 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for MA200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ma50_aligned[i]) or np.isnan(ma200_aligned[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Golden Cross: MA50 crosses above MA200 + volume confirmation
            if ma50_aligned[i] > ma200_aligned[i] and ma50_aligned[i-1] <= ma200_aligned[i-1] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Death Cross: MA50 crosses below MA200 + volume confirmation
            elif ma50_aligned[i] < ma200_aligned[i] and ma50_aligned[i-1] >= ma200_aligned[i-1] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit on Death Cross
            if ma50_aligned[i] < ma200_aligned[i] and ma50_aligned[i-1] >= ma200_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit on Golden Cross
            if ma50_aligned[i] > ma200_aligned[i] and ma50_aligned[i-1] <= ma200_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals