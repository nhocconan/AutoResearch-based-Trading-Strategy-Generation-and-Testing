#!/usr/bin/env python3
"""
6h_Chaikin_Money_Flow_Zone_v1
Hypothesis: Combines CMF (money flow) with Bollinger Band width to identify
accumulation/distribution zones. In both bull and bear markets, extreme money
flow coinciding with low volatility (squeeze) precedes powerful moves. Uses 1d
trend filter to align with higher timeframe momentum.
Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
"""

name = "6h_Chaikin_Money_Flow_Zone_v1"
timeframe = "6h"
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
    
    # === 1D Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Bollinger Band Width (20,2) on 6h ===
    bb_length = 20
    bb_mult = 2.0
    
    # Basis (SMA)
    basis = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    # Deviation
    dev = bb_mult * pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    upper = basis + dev
    lower = basis - dev
    # BB Width as percentage of basis
    bb_width = np.where(basis != 0, (upper - lower) / basis, 0)
    
    # === Chaikin Money Flow (20) ===
    # Money Flow Multiplier
    mfm = np.where((high - low) != 0, ((close - low) - (high - close)) / (high - low), 0)
    # Money Flow Volume
    mfv = mfm * volume
    # CMF
    cmf_length = 20
    mfv_sum = pd.Series(mfv).rolling(window=cmf_length, min_periods=cmf_length).sum().values
    volume_sum = pd.Series(volume).rolling(window=cmf_length, min_periods=cmf_length).sum().values
    cmf = np.where(volume_sum != 0, mfv_sum / volume_sum, 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(bb_length, cmf_length, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(bb_width[i]) or 
            np.isnan(cmf[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: CMF > 0.15 (strong buying pressure) + BB width < 0.05 (squeeze)
            if cmf[i] > 0.15 and bb_width[i] < 0.05 and ema50_1d_aligned[i] < close[i]:
                signals[i] = 0.25
                position = 1
            # Short: CMF < -0.15 (strong selling pressure) + BB width < 0.05 (squeeze)
            elif cmf[i] < -0.15 and bb_width[i] < 0.05 and ema50_1d_aligned[i] > close[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: CMF turns negative or BB width expands significantly
            if cmf[i] < 0 or bb_width[i] > 0.15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: CMF turns positive or BB width expands significantly
            if cmf[i] > 0 or bb_width[i] > 0.15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals