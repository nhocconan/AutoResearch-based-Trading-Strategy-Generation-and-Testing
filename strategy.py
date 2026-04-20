# 4h_1d_ChaikinMoneyFlow_Signal
# Hypothesis: Use 1-day Chaikin Money Flow (CMF) as a trend filter on 4h timeframe, with price crossing above/below 4h VWAP as entry triggers.
# CMF > 0 indicates buying pressure (bullish), CMF < 0 indicates selling pressure (bearish).
# VWAP crossovers provide entry timing aligned with intraday momentum.
# Works in both bull and bear markets by following institutional money flow direction.
# Target: 20-40 trades per year with low turnover to minimize fee drag.

name = "4h_1d_ChaikinMoneyFlow_Signal"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Chaikin Money Flow (CMF) - 20 period
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Money Flow Multiplier
    mfm = ((close_1d - low_1d) - (high_1d - close_1d)) / (high_1d - low_1d)
    mfm = np.where((high_1d - low_1d) == 0, 0, mfm)  # Avoid division by zero
    
    # Money Flow Volume
    mfv = mfm * volume_1d
    
    # 20-period CMF
    cmf = np.zeros_like(close_1d)
    for i in range(20, len(close_1d)):
        cmf[i] = np.sum(mfv[i-19:i+1]) / np.sum(volume_1d[i-19:i+1]) if np.sum(volume_1d[i-19:i+1]) > 0 else 0
    
    # Align 1d CMF to 4h timeframe
    cmf_aligned = align_htf_to_ltf(prices, df_1d, cmf)
    
    # Calculate 4h VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator > 0, vwap_numerator / vwap_denominator, 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(cmf_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: CMF > 0 (bullish pressure) and price crosses above VWAP
            if cmf_aligned[i] > 0 and close[i] > vwap[i] and (i == 0 or close[i-1] <= vwap[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: CMF < 0 (bearish pressure) and price crosses below VWAP
            elif cmf_aligned[i] < 0 and close[i] < vwap[i] and (i == 0 or close[i-1] >= vwap[i-1]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: CMF turns negative or price crosses below VWAP
            if cmf_aligned[i] < 0 or (close[i] < vwap[i] and close[i-1] >= vwap[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CMF turns positive or price crosses above VWAP
            if cmf_aligned[i] > 0 or (close[i] > vwap[i] and close[i-1] <= vwap[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals