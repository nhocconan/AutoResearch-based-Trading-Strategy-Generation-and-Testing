#!/usr/bin/env python3
# 12H_POWER_INDEX_CROSSOVER_1D_TREND_FILTER
# Hypothesis: Power Index (bull power minus bear power) on daily timeframe
# measures net institutional pressure. Cross above/below zero with trend filter
# (price relative to daily EMA34) captures momentum shifts in both bull and bear markets.
# Uses 12h timeframe for lower frequency trading to reduce fee drag.
# Target: 12-37 trades/year (50-150 total over 4 years) on 12h timeframe.

name = "12H_POWER_INDEX_CROSSOVER_1D_TREND_FILTER"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Daily data for Power Index calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 for Power Index calculation
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = EMA13 - Low (buying pressure)
    # Bear Power = High - EMA13 (selling pressure)
    # Power Index = Bull Power - Bear Power = 2*EMA13 - (High + Low)
    power_index = 2 * ema13 - (high_1d + low_1d)
    
    # EMA34 for trend filter
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 12h timeframe
    power_index_aligned = align_htf_to_ltf(prices, df_1d, power_index)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least one day of data
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(power_index_aligned[i]) or np.isnan(ema34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Power Index crosses above zero in uptrend
            if (power_index_aligned[i] > 0 and 
                power_index_aligned[i-1] <= 0 and
                close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Power Index crosses below zero in downtrend
            elif (power_index_aligned[i] < 0 and 
                  power_index_aligned[i-1] >= 0 and
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Power Index crosses below zero or trend reversal
            if (power_index_aligned[i] < 0 and 
                power_index_aligned[i-1] >= 0) or \
               close[i] <= ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Power Index crosses above zero or trend reversal
            if (power_index_aligned[i] > 0 and 
                power_index_aligned[i-1] <= 0) or \
               close[i] >= ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals