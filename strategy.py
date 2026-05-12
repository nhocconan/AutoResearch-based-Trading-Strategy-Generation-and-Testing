#!/usr/bin/env python3
name = "6h_ChaikinMoneyFlow_Trend_1dVWAP_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # === 1D DATA FOR VWAP FILTER ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # === VWAP CALCULATION (1d) ===
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_numerator = np.cumsum(typical_price_1d * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap_1d = vwap_numerator / vwap_denominator
    
    # === CHAIKIN MONEY FLOW (20-period on 6h) ===
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = np.where(high == low, 0, mfm)  # avoid division by zero
    mfv = mfm * volume
    cmf = pd.Series(mfv).rolling(window=20, min_periods=20).sum() / \
          pd.Series(volume).rolling(window=20, min_periods=20).sum()
    cmf_values = cmf.values
    
    # === ALIGN 1D VWAP TO 6H TIMEFRAME ===
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(cmf_values[i]) or 
            np.isnan(vwap_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: CMF > 0.1 + PRICE ABOVE VWAP
            if (cmf_values[i] > 0.1 and 
                close[i] > vwap_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: CMF < -0.1 + PRICE BELOW VWAP
            elif (cmf_values[i] < -0.1 and 
                  close[i] < vwap_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: CMF < 0 OR PRICE BELOW VWAP
            if (cmf_values[i] < 0 or 
                close[i] < vwap_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CMF > 0 OR PRICE ABOVE VWAP
            if (cmf_values[i] > 0 or 
                close[i] > vwap_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals