#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_camarilla_pivot_reversal_v2
# Uses 1d Camarilla pivot levels with volume confirmation and 1w trend filter.
# Longs near S1/S2 (mean reversion) in weekly uptrend, shorts near R1/R2 in weekly downtrend.
# Volume spike confirms institutional interest at key levels. Target: 20-50 trades/year.
name = "6h_1d_camarilla_pivot_reversal_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    R1 = pivot_1d + range_1d * 1.1 / 12
    R2 = pivot_1d + range_1d * 1.1 / 6
    R3 = pivot_1d + range_1d * 1.1 / 4
    R4 = pivot_1d + range_1d * 1.1 / 2
    S1 = pivot_1d - range_1d * 1.1 / 12
    S2 = pivot_1d - range_1d * 1.1 / 6
    S3 = pivot_1d - range_1d * 1.1 / 4
    S4 = pivot_1d - range_1d * 1.1 / 2
    
    # Align 1d Camarilla levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 1w EMA10 for trend filter
    close_1w = df_1w['close'].values
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Volume filter: 1.5x average volume
    vol_ma = np.zeros_like(volume)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(R2_aligned[i]) or
            np.isnan(S1_aligned[i]) or np.isnan(S2_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(ema_10_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume spike condition
        vol_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Determine weekly trend
        weekly_uptrend = close[i] > ema_10_1w_aligned[i]
        weekly_downtrend = close[i] < ema_10_1w_aligned[i]
        
        # Long conditions: price near S1/S2 in weekly uptrend with volume spike
        near_support = (low[i] <= S2_aligned[i] * 1.002) and (low[i] >= S1_aligned[i] * 0.998)
        long_condition = weekly_uptrend and near_support and vol_spike
        
        # Short conditions: price near R1/R2 in weekly downtrend with volume spike
        near_resistance = (high[i] >= R1_aligned[i] * 0.998) and (high[i] <= R2_aligned[i] * 1.002)
        short_condition = weekly_downtrend and near_resistance and vol_spike
        
        # Exit conditions: opposite touch or weekly trend change
        exit_long = (high[i] >= R1_aligned[i] * 0.998) or (weekly_downtrend and close[i] < ema_10_1w_aligned[i])
        exit_short = (low[i] <= S1_aligned[i] * 1.002) or (weekly_uptrend and close[i] > ema_10_1w_aligned[i])
        
        if long_condition and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_condition and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals