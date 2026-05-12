#!/usr/bin/env python3
# 4H_CAMARILLA_R4_S4_BREAKOUT_1D_TREND_FILTER
# Hypothesis: Daily Camarilla R4/S4 levels (extreme levels) act as stronger support/resistance than R3/S3 on 4h timeframe.
# Breakouts above R4 or below S4 with daily trend filter (EMA34) capture momentum while reducing false breaks.
# Works in bull markets (breakouts continuation) and bear markets (reversals at extremes).
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years).

name = "4H_CAMARILLA_R4_S4_BREAKOUT_1D_TREND_FILTER"
timeframe = "4h"
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
    
    # Daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # R4 = C + (H-L)*1.1, S4 = C - (H-L)*1.1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r4 = close_1d + (high_1d - low_1d) * 1.1
    s4 = close_1d - (high_1d - low_1d) * 1.1
    
    # EMA34 for trend filter
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least one day of data
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(ema34_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R4 in uptrend
            if (close[i] > r4_aligned[i] and 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S4 in downtrend
            elif (close[i] < s4_aligned[i] and 
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below S4 or trend reversal
            if (close[i] < s4_aligned[i] or 
                close[i] <= ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above R4 or trend reversal
            if (close[i] > r4_aligned[i] or 
                close[i] >= ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals