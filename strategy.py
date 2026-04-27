#!/usr/bin/env python3
"""
6h_PriceAction_Reversal_at_DailyPivots_VolumeFilter
Hypothesis: Reversal at daily pivot points (PP, R1, S1) with volume confirmation works in both bull and bear markets by capturing mean reversion at key intraday support/resistance levels. Uses 6h timeframe for lower frequency and 1d pivot levels for structure. Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Standard pivot point calculation
    PP = (high_1d + low_1d + close_1d) / 3.0
    R1 = 2 * PP - low_1d
    S1 = 2 * PP - high_1d
    R2 = PP + (high_1d - low_1d)
    S2 = PP - (high_1d - low_1d)
    
    # Align to 6h timeframe (previous day's pivots available at open)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)  # Moderate volume spike
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(PP_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(R2_aligned[i]) or 
            np.isnan(S2_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price touches or crosses below S1 with volume spike, reverse back above S1
            if (low[i] <= S1_aligned[i] and close[i] > S1_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches or crosses above R1 with volume spike, reverse back below R1
            elif (high[i] >= R1_aligned[i] and close[i] < R1_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price reaches R1 or R2 (take profit) or breaks below S2 (stop)
            if (close[i] >= R1_aligned[i] or 
                close[i] <= S2_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches S1 or S2 (take profit) or breaks above R2 (stop)
            if (close[i] <= S1_aligned[i] or 
                close[i] >= R2_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_PriceAction_Reversal_at_DailyPivots_VolumeFilter"
timeframe = "6h"
leverage = 1.0