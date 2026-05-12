#!/usr/bin/env python3
# 1D_CAMARILLA_R3_S3_BREAKOUT_1W_TREND_FILTER
# Hypothesis: Camarilla pivot levels (R3/S3) from weekly chart act as strong support/resistance.
# Breakouts above R3 or below S3 with weekly trend filter capture momentum moves.
# Works in bull markets (breakouts continuation) and bear markets (reversals at extremes).
# Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years).

name = "1D_CAMARILLA_R3_S3_BREAKOUT_1W_TREND_FILTER"
timeframe = "1d"
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
    
    # Weekly data for Camarilla calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate R3 and S3 for each week
    r3 = close_1w + (high_1w - low_1w) * 1.25 / 2
    s3 = close_1w - (high_1w - low_1w) * 1.25 / 2
    
    # EMA34 for weekly trend filter
    ema34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 1d timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    ema34_aligned = align_htf_to_ltf(prices, df_1w, ema34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least one week of data
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema34_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 in uptrend
            if (close[i] > r3_aligned[i] and 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 in downtrend
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below S3 or trend reversal
            if (close[i] < s3_aligned[i] or 
                close[i] <= ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above R3 or trend reversal
            if (close[i] > r3_aligned[i] or 
                close[i] >= ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals