#!/usr/bin/env python3
"""
1d_1w_MeanReversion_ZScore
Hypothesis: Weekly mean reversion of daily price deviations works across bull and bear markets.
When daily price deviates significantly from weekly mean (Z-score > 2 or < -2), price tends to revert.
Volume confirmation reduces false signals. Works on BTC/ETH with low trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data (HTF)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly mean and standard deviation (20-period)
    weekly_mean = np.full(len(close_1w), np.nan)
    weekly_std = np.full(len(close_1w), np.nan)
    
    for i in range(20, len(close_1w)):
        weekly_mean[i] = np.mean(close_1w[i-20:i])
        weekly_std[i] = np.std(close_1w[i-20:i])
    
    # Z-score of current daily close relative to weekly distribution
    zscore = np.full(len(close_1w), np.nan)
    for i in range(20, len(close_1w)):
        if weekly_std[i] > 0:
            zscore[i] = (close_1w[i] - weekly_mean[i]) / weekly_std[i]
    
    # Align weekly Z-score to daily timeframe
    zscore_aligned = align_htf_to_ltf(prices, df_1w, zscore)
    
    # Volume spike: current volume > 2.0 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure weekly indicators ready
    
    for i in range(start_idx, n):
        if np.isnan(zscore_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price significantly below weekly mean (Z < -2) with volume spike
            if zscore_aligned[i] < -2.0 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price significantly above weekly mean (Z > 2) with volume spike
            elif zscore_aligned[i] > 2.0 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to weekly mean (Z > -0.5) or extreme reversal
            if zscore_aligned[i] > -0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly mean (Z < 0.5) or extreme reversal
            if zscore_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_MeanReversion_ZScore"
timeframe = "1d"
leverage = 1.0