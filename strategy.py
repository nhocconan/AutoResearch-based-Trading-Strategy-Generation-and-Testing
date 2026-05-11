#!/usr/bin/env python3
name = "6h_WeeklyPivot_PriceAction_Volume"
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
    
    # 1d data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Weekly pivot from previous week
    # Using daily data to calculate weekly pivot: (H+L+C)/3
    high_wk = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    low_wk = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    close_wk = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    pivot = (high_wk + low_wk + close_wk) / 3.0
    # S1 and R1 levels
    s1 = 2 * pivot - high_wk
    r1 = 2 * pivot - low_wk
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Volume filter: 6h volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    # Price action: higher high/low for trend
    hh = high > pd.Series(high).rolling(window=2, min_periods=2).max().shift(1).values
    ll = low < pd.Series(low).rolling(window=2, min_periods=2).min().shift(1).values
    # Align to avoid look-ahead
    hh_aligned = pd.Series(hh).shift(1).fillna(False).values
    ll_aligned = pd.Series(ll).shift(1).fillna(False).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA and pivots
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above pivot + making higher low + volume filter
            if close[i] > pivot_aligned[i] and ll_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below pivot + making lower high + volume filter
            elif close[i] < pivot_aligned[i] and hh_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price below S1 or making lower high
            if close[i] < s1_aligned[i] or hh_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price above R1 or making higher low
            if close[i] > r1_aligned[i] or ll_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals