#!/usr/bin/env python3
# 1H_CAMARILLA_R1_S1_BREAKOUT_4H_TREND_FILTER
# Hypothesis: Hourly price breaks above R1 or below S1 with 4-hour EMA20 trend filter capture momentum.
# Uses 4-hour trend for direction (reduces whipsaw), 1h for entry timing.
# Works in bull markets (breakouts continuation) and bear markets (reversals at extremes).
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.
# Session filter: 08-20 UTC to reduce noise.

name = "1H_CAMARILLA_R1_S1_BREAKOUT_4H_TREND_FILTER"
timeframe = "1h"
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
    
    # 4-hour data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # EMA20 for 4h trend filter
    ema20 = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_aligned = align_htf_to_ltf(prices, df_4h, ema20)
    
    # 1-day data for Camarilla calculation (more stable levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day: R1 = C + (H-L)*1.125/6, S1 = C - (H-L)*1.125/6
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r1 = close_1d + (high_1d - low_1d) * 1.125 / 6
    s1 = close_1d - (high_1d - low_1d) * 1.125 / 6
    
    # Align to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least one day of data
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 in uptrend (price > EMA20)
            if (close[i] > r1_aligned[i] and 
                close[i] > ema20_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 in downtrend (price < EMA20)
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema20_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below S1 or trend reversal (price <= EMA20)
            if (close[i] < s1_aligned[i] or 
                close[i] <= ema20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price rises above R1 or trend reversal (price >= EMA20)
            if (close[i] > r1_aligned[i] or 
                close[i] >= ema20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals