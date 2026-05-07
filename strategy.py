#!/usr/bin/env python3
"""
12h_1wCrossover_1dVolumeFilter_v1
Hypothesis: Uses weekly EMA crossover for primary trend direction, confirmed by
daily volume expansion, on 12h timeframe. Targets 15-25 trades/year to minimize
fee drag. Works in bull markets via trend continuation and in bear markets via
mean-reversion bounces off the weekly EMA with volume confirmation.
"""

name = "12h_1wCrossover_1dVolumeFilter_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA crossover (fast=21, slow=55)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_fast = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_slow = pd.Series(close_1w).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Weekly trend: 1 if fast > slow, -1 if fast < slow
    weekly_trend = np.where(ema_fast > ema_slow, 1, -1)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # Daily volume confirmation: current day volume > 20-day average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(55, n):
        # Skip if any critical value is NaN
        if (np.isnan(weekly_trend_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or vol_ma_1d_aligned[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current day's volume (aligned to 12h)
        vol_today = vol_ma_1d_aligned[i]
        
        if position == 0:
            # Enter long: weekly uptrend AND above-average volume
            if weekly_trend_aligned[i] == 1 and volume[i] > vol_today:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly downtrend AND above-average volume
            elif weekly_trend_aligned[i] == -1 and volume[i] > vol_today:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: weekly trend turns down
            if weekly_trend_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: weekly trend turns up
            if weekly_trend_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals