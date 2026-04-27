#!/usr/bin/env python3
"""
6h_ChaikinMoneyFlow_WeeklyTrend
Hypothesis: 6h Chaikin Money Flow (CMF) with weekly trend filter for institutional flow confirmation.
- CMF(20) > 0 indicates buying pressure, < 0 selling pressure on 6h
- Weekly trend: price > weekly EMA50 for longs, < for shorts
- Entry: CMF crosses above 0.1 + weekly uptrend (long), CMF crosses below -0.1 + weekly downtrend (short)
- Exit: CMF crosses back through 0 or trend failure
- Designed to capture smart money flows while avoiding counter-trend moves
- Target: 20-35 trades/year on 6h (80-140 total over 4 years)
"""

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
    
    # 6h Chaikin Money Flow (CMF) - 20 period
    # CMF = Sum((Close - Low - (High - Close)) / (High - Low) * Volume) / Sum(Volume)
    mfm = np.zeros_like(close)
    mfv = np.zeros_like(close)
    
    # Avoid division by zero
    hl_range = high - low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    
    mfm = ((close - low) - (high - close)) / hl_range
    mfv = mfm * volume
    
    # Calculate 20-period sums
    cmf = np.full(n, np.nan)
    for i in range(20, n):
        sum_mfv = np.sum(mfv[i-20:i+1])
        sum_vol = np.sum(volume[i-20:i+1])
        if sum_vol != 0:
            cmf[i] = sum_mfv / sum_vol
    
    # Weekly trend filter (EMA50)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    close_weekly = df_weekly['close'].values
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for calculations
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(cmf[i]) or np.isnan(ema50_weekly_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: CMF crosses above 0.1 + weekly uptrend
            if (cmf[i] > 0.1 and cmf[i-1] <= 0.1 and close[i] > ema50_weekly_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: CMF crosses below -0.1 + weekly downtrend
            elif (cmf[i] < -0.1 and cmf[i-1] >= -0.1 and close[i] < ema50_weekly_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: CMF crosses below 0 or weekly trend failure
            if (cmf[i] < 0 or close[i] < ema50_weekly_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: CMF crosses above 0 or weekly trend failure
            if (cmf[i] > 0 or close[i] > ema50_weekly_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ChaikinMoneyFlow_WeeklyTrend"
timeframe = "6h"
leverage = 1.0