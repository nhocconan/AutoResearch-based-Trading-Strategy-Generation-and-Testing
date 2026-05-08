#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Camarilla_Pivot_Trend_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot calculation
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla levels: R3, R1, S1, S3
    r3_1w = pivot_1w + (range_1w * 1.1)
    r1_1w = pivot_1w + (range_1w * 1.1 / 6)
    s1_1w = pivot_1w - (range_1w * 1.1 / 6)
    s3_1w = pivot_1w - (range_1w * 1.1)
    
    # Align weekly levels to daily timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Calculate daily EMA(34) for trend filter
    daily_ema34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily volume spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for weekly alignment
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(daily_ema34[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price touches S1 with volume spike and above weekly EMA
            if (low[i] <= s1_1w_aligned[i] and volume_spike[i] and 
                close[i] > daily_ema34[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price touches R1 with volume spike and below weekly EMA
            elif (high[i] >= r1_1w_aligned[i] and volume_spike[i] and 
                  close[i] < daily_ema34[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches S3 or closes below daily EMA
            if (low[i] <= s3_1w_aligned[i] or close[i] < daily_ema34[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches R3 or closes above daily EMA
            if (high[i] >= r3_1w_aligned[i] or close[i] > daily_ema34[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals