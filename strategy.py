#!/usr/bin/env python3
"""
6h_Chaikin_Money_Flow_Signal_Line
Hypothesis: Chaikin Money Flow (CMF) with signal line crossover on 6h timeframe
CMF > 0 indicates buying pressure, CMF < 0 indicates selling pressure
Signal line (EMA of CMF) provides smoothed momentum signal
Works in bull/bear via zero-line cross with momentum confirmation
Targets 50-150 total trades over 4 years (12-37/year) with low turnover
"""

name = "6h_Chaikin_Money_Flow_Signal_Line"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Money Flow Multiplier
    # Avoid division by zero when high == low
    hl_range = high - low
    mfm = np.where(hl_range != 0, ((close - low) - (high - close)) / hl_range, 0)
    
    # Money Flow Volume
    mfv = mfm * volume
    
    # Chaikin Money Flow (20-period)
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    volume_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = np.where(volume_sum != 0, mfv_sum / volume_sum, 0)
    
    # Signal line: EMA of CMF (9-period)
    cmf_series = pd.Series(cmf)
    signal_line = cmf_series.ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Histogram: CMF - Signal Line
    histogram = cmf - signal_line
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for CMF calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(cmf[i]) or np.isnan(signal_line[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: CMF crosses above signal line with positive momentum
            if cmf[i] > signal_line[i] and cmf[i-1] <= signal_line[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: CMF crosses below signal line with negative momentum
            elif cmf[i] < signal_line[i] and cmf[i-1] >= signal_line[i-1]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if CMF crosses below signal line
            if cmf[i] < signal_line[i] and cmf[i-1] >= signal_line[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if CMF crosses above signal line
            if cmf[i] > signal_line[i] and cmf[i-1] <= signal_line[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals