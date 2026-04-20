#!/usr/bin/env python3
"""
4h_1d_ChaikinMoneyFlow_Volume_Trend_v1
Concept: 4h Chaikin Money Flow with 1d trend filter and volume confirmation.
- Long: CMF(20) > 0 AND price > 1d EMA200 AND volume > 1.5x average volume
- Short: CMF(20) < 0 AND price < 1d EMA200 AND volume > 1.5x average volume
- Exit: CMF crosses zero
- Position sizing: 0.25
- Uses volume confirmation to avoid false signals and reduce overtrading
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_ChaikinMoneyFlow_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 210:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # === 4h: Chaikin Money Flow (20) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = np.where((high - low) == 0, 0, mfm)  # Avoid division by zero
    
    # Money Flow Volume
    mfv = mfm * volume
    
    # CMF = 20-period sum of MFV / 20-period sum of volume
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    volume_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = np.where(volume_sum != 0, mfv_sum / volume_sum, 0)
    
    # === 4h: Volume confirmation (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    # === Daily: EMA200 trend filter ===
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for CMF calculation
    
    for i in range(start_idx, n):
        # Get values
        cmf_val = cmf[i]
        vol_val = volume[i]
        vol_thresh_val = vol_threshold[i]
        ema200_val = ema200_1d_aligned[i]
        
        # Skip if any value is NaN or invalid
        if (np.isnan(cmf_val) or np.isnan(vol_val) or np.isnan(vol_thresh_val) or 
            np.isnan(ema200_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Positive CMF, price above 1d EMA200, and volume confirmation
            if cmf_val > 0 and close[i] > ema200_val and vol_val > vol_thresh_val:
                signals[i] = 0.25
                position = 1
            # Short: Negative CMF, price below 1d EMA200, and volume confirmation
            elif cmf_val < 0 and close[i] < ema200_val and vol_val > vol_thresh_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: CMF crosses below zero
            if cmf_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CMF crosses above zero
            if cmf_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals