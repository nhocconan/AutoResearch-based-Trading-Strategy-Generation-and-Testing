#!/usr/bin/env python3
"""
6h_Chaikin_Money_Flow_Energy_1dTrend_Confirmation_v1
Hypothesis: Chaikin Money Flow (CMF) measures buying/selling pressure via volume-weighted accumulation/distribution.
Long when CMF > 0.15 + price > 1d EMA50 (uptrend); short when CMF < -0.15 + price < 1d EMA50 (downtrend).
Uses 1d EMA50 trend filter to avoid counter-trend trades. Designed for 6h timeframe to target 12-37 trades/year.
Works in both bull (follows strong money flow) and bear (avoids false signals via trend filter).
"""

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Money Flow Multiplier: [(Close - Low) - (High - Close)] / (High - Low)
    # Avoid division by zero
    hl_range = high - low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)  # small epsilon to prevent div/0
    mfm = ((close - low) - (high - close)) / hl_range
    
    # Money Flow Volume = Money Flow Multiplier * Volume
    mfv = mfm * volume
    
    # Chaikin Money Flow (20-period sum of MFV / 20-period sum of volume)
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    vol_sum = np.where(vol_sum == 0, 1e-10, vol_sum)  # prevent div/0
    cmf = mfv_sum / vol_sum
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need EMA50 (50) and CMF (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(cmf[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema50 = ema50_1d_aligned[i]
        cmf_val = cmf[i]
        
        if position == 0:
            # Long: CMF > 0.15 (buying pressure) + uptrend (price > EMA50)
            if cmf_val > 0.15 and close_val > ema50:
                signals[i] = size
                position = 1
                entry_price = close_val
            # Short: CMF < -0.15 (selling pressure) + downtrend (price < EMA50)
            elif cmf_val < -0.15 and close_val < ema50:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: CMF turns negative (< 0) or price breaks below EMA50
            if cmf_val < 0 or close_val < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: CMF turns positive (> 0) or price breaks above EMA50
            if cmf_val > 0 or close_val > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Chaikin_Money_Flow_Energy_1dTrend_Confirmation_v1"
timeframe = "6h"
leverage = 1.0