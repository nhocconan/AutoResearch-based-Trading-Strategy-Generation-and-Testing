#!/usr/bin/env python3
# 12h_ChaikinMoneyFlow_BullBear_Trend_Filter
# Hypothesis: Chaikin Money Flow (CMF) on 1d shows institutional accumulation/distribution.
# In trending markets (price above/below 1d EMA50), CMF extremes confirm trend strength.
# Enter long when price > EMA50 and CMF > +0.15; short when price < EMA50 and CMF < -0.15.
# Exit when trend fails or CMF reverts toward zero. Works in bull/bear by following 1d trend.
# Uses 12h timeframe for entries, targeting 50-150 trades over 4 years.

name = "12h_ChaikinMoneyFlow_BullBear_Trend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for CMF and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Chaikin Money Flow (CMF) over 20 days
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Money Flow Volume = Money Flow Multiplier * Volume
    # CMF = 20-period sum of Money Flow Volume / 20-period sum of Volume
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    mfm = np.where((high_1d - low_1d) != 0, 
                   ((close_1d - low_1d) - (high_1d - close_1d)) / (high_1d - low_1d), 
                   0.0)
    mfv = mfm * volume_1d
    
    cmf_20 = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values / \
             pd.Series(volume_1d).rolling(window=20, min_periods=20).sum().values
    cmf_20_aligned = align_htf_to_ltf(prices, df_1d, cmf_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50), CMF (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(cmf_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: uptrend + CMF > +0.15 (strong buying pressure)
            if uptrend and cmf_20_aligned[i] > 0.15:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + CMF < -0.15 (strong selling pressure)
            elif downtrend and cmf_20_aligned[i] < -0.15:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend fails OR CMF drops below +0.05 (weakening buying)
            if not uptrend or cmf_20_aligned[i] < 0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend fails OR CMF rises above -0.05 (weakening selling)
            if not downtrend or cmf_20_aligned[i] > -0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals