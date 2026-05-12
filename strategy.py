#!/usr/bin/env python3
# 6H_WEEKLY_PIVOT_DIRECTION_1D_VOLUME_FILTER
# Hypothesis: Weekly pivot points (from prior week) define major support/resistance.
# Breakouts above weekly R1 or below weekly S1 with daily trend filter (EMA34) and volume spike capture momentum.
# Works in bull markets (breakouts continuation) and bear markets (reversals at extremes).
# Target: 15-30 trades/year on 6h timeframe (60-120 total over 4 years).

name = "6H_WEEKLY_PIVOT_DIRECTION_1D_VOLUME_FILTER"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    # Standard pivot: P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    
    # Daily data for trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # EMA34 for trend filter
    ema34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume ratio: current volume / 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma, out=np.ones_like(volume), where=vol_ma!=0)
    
    # Align to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least one week of data
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema34_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 in uptrend with volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema34_aligned[i] and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 in downtrend with volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema34_aligned[i] and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below S1 or trend reversal
            if (close[i] < s1_aligned[i] or 
                close[i] <= ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above R1 or trend reversal
            if (close[i] > r1_aligned[i] or 
                close[i] >= ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals