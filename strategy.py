#!/usr/bin/env python3
# 6H_1D_12H_ChaikinMoneyFlow_Trend_Filter
# Hypothesis: Chaikin Money Flow (CMF) on daily timeframe measures institutional buying/selling pressure.
# In trending markets, CMF > 0 indicates accumulation (bullish), CMF < 0 indicates distribution (bearish).
# Entry: Long when price breaks above 12h EMA(20) AND daily CMF > 0.05 (strong accumulation)
#        Short when price breaks below 12h EMA(20) AND daily CMF < -0.05 (strong distribution)
# Exit: Reverse signal or trend change (EMA crossover)
# Uses 12h EMA for trend and entry timing, daily CMF for institutional bias.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years) to avoid fee drag.

name = "6H_1D_12H_ChaikinMoneyFlow_Trend_Filter"
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
    
    # Get daily data for CMF calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Chaikin Money Flow (CMF) over 20 days
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # = [(Close - Low) - (High - Close)] / (High - Low)
    # = [2*Close - High - Low] / (High - Low)
    mfm = ((2 * close_1d - high_1d - low_1d) / (high_1d - low_1d))
    # Handle division by zero (when high == low)
    mfm = np.where((high_1d - low_1d) == 0, 0, mfm)
    
    # Money Flow Volume = MFM * Volume
    mfv = mfm * volume_1d
    
    # 20-period CMF = Sum(MFV, 20) / Sum(Volume, 20)
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume_1d).rolling(window=20, min_periods=20).sum().values
    cmf = np.where(vol_sum != 0, mfv_sum / vol_sum, 0)
    
    # Get 12h data for EMA(20) trend filter and entry timing
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_20 = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align daily CMF and 12h EMA to 6h timeframe
    cmf_aligned = align_htf_to_ltf(prices, df_1d, cmf)
    ema_20_aligned = align_htf_to_ltf(prices, df_12h, ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(cmf_aligned[i]) or np.isnan(ema_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above 12h EMA(20) AND strong accumulation (CMF > 0.05)
            if close[i] > ema_20_aligned[i] and cmf_aligned[i] > 0.05:
                signals[i] = 0.25
                position = 1
            # Enter short: price below 12h EMA(20) AND strong distribution (CMF < -0.05)
            elif close[i] < ema_20_aligned[i] and cmf_aligned[i] < -0.05:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below EMA OR CMF turns negative (distribution)
            if close[i] < ema_20_aligned[i] or cmf_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above EMA OR CMF turns positive (accumulation)
            if close[i] > ema_20_aligned[i] or cmf_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals