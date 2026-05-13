#!/usr/bin/env python3
"""
6h_Chaikin_Money_Flow_With_1d_Trend_Filter
Hypothesis: Chaikin Money Flow (CMF) measures buying/selling pressure via volume-weighted accumulation/distribution. 
Combined with 1d trend filter (price above/below 50-period EMA) to ensure trades align with higher timeframe direction.
Works in both bull and bear markets by following institutional money flow. Target: 20-40 trades/year per symbol.
"""

name = "6h_Chaikin_Money_Flow_With_1d_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Chaikin Money Flow (20-period)
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Money Flow Volume = Money Flow Multiplier * Volume
    # CMF = 20-period sum of Money Flow Volume / 20-period sum of Volume
    high_low = high - low
    # Avoid division by zero
    high_low_safe = np.where(high_low == 0, 1e-10, high_low)
    mfm = ((close - low) - (high - close)) / high_low_safe
    mfv = mfm * volume
    
    # 20-period sums
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = mfv_sum / vol_sum  # Range: -1 to +1
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        cmf_val = cmf[i]
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        
        if position == 0:
            # LONG: CMF > 0.05 (buying pressure) and 1d uptrend
            if cmf_val > 0.05 and uptrend:
                signals[i] = 0.25
                position = 1
            # SHORT: CMF < -0.05 (selling pressure) and 1d downtrend
            elif cmf_val < -0.05 and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: CMF < 0 (loss of buying pressure) or trend fails
            if cmf_val < 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CMF > 0 (loss of selling pressure) or trend fails
            if cmf_val > 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals