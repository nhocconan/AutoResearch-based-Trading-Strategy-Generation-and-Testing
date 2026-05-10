#!/usr/bin/env python3
# 1H_4H_1D_Camarilla_R1_S1_Breakout_Trend_Filter
# Hypothesis: Use 1d Camarilla pivot structure for trend direction and 4h EMA50 filter.
# Enter on 1h breakouts of daily R1/S1 levels only when 4h EMA50 confirms trend.
# Target: 15-35 trades/year per symbol (60-140 total over 4 years) to avoid fee drag.

name = "1H_4H_1D_Camarilla_R1_S1_Breakout_Trend_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar using previous day's range
    H4 = np.full_like(close_1d, np.nan)
    L4 = np.full_like(close_1d, np.nan)
    H3 = np.full_like(close_1d, np.nan)
    L3 = np.full_like(close_1d, np.nan)
    H2 = np.full_like(close_1d, np.nan)
    L2 = np.full_like(close_1d, np.nan)
    H1 = np.full_like(close_1d, np.nan)
    L1 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's high, low, close
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        
        if range_val <= 0:
            continue
            
        H4[i] = prev_close + range_val * 1.1 / 2
        L4[i] = prev_close - range_val * 1.1 / 2
        H3[i] = prev_close + range_val * 1.1 / 4
        L3[i] = prev_close - range_val * 1.1 / 4
        H2[i] = prev_close + range_val * 1.1 / 6
        L2[i] = prev_close - range_val * 1.1 / 6
        H1[i] = prev_close + range_val * 1.1 / 12
        L1[i] = prev_close - range_val * 1.1 / 12
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 1h
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H2_aligned = align_htf_to_ltf(prices, df_1d, H2)
    L2_aligned = align_htf_to_ltf(prices, df_1d, L2)
    H1_aligned = align_htf_to_ltf(prices, df_1d, H1)
    L1_aligned = align_htf_to_ltf(prices, df_1d, L1)
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H1_aligned[i]) or np.isnan(L1_aligned[i]) or 
            np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(ema_4h_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 4h EMA50
        uptrend = close[i] > ema_4h_50_aligned[i]
        downtrend = close[i] < ema_4h_50_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above H1 with uptrend
            if close[i] > H1_aligned[i] and uptrend:
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below L1 with downtrend
            elif close[i] < L1_aligned[i] and downtrend:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below L1 or trend changes
            if close[i] < L1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price breaks above H1 or trend changes
            if close[i] > H1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals