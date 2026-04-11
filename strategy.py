#!/usr/bin/env python3
# 4h_1d_camarilla_pivot_v1
# Strategy: 4h Camarilla pivot breakout with 1d EMA trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla levels from daily chart act as strong support/resistance.
# Breakouts above/below these levels with volume confirmation and 1d EMA trend alignment
# yield high-probability trades. Works in both bull (breakouts continue) and bear
# (breakouts fail, mean revert to opposite level) regimes via symmetric logic.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_v1"
timeframe = "4h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close
    close_val = close
    L4 = close_val + range_val * 1.1 / 2
    L3 = close_val + range_val * 1.1 / 4
    L2 = close_val + range_val * 1.1 / 6
    L1 = close_val + range_val * 1.1 / 12
    S1 = close_val - range_val * 1.1 / 12
    S2 = close_val - range_val * 1.1 / 6
    S3 = close_val - range_val * 1.1 / 4
    S4 = close_val - range_val * 1.1 / 2
    return L4, L3, L2, L1, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous day
    L4_arr = np.full(n, np.nan)
    L3_arr = np.full(n, np.nan)
    L2_arr = np.full(n, np.nan)
    L1_arr = np.full(n, np.nan)
    S1_arr = np.full(n, np.nan)
    S2_arr = np.full(n, np.nan)
    S3_arr = np.full(n, np.nan)
    S4_arr = np.full(n, np.nan)
    
    for i in range(len(df_1d)):
        day_high = df_1d['high'].iloc[i]
        day_low = df_1d['low'].iloc[i]
        day_close = df_1d['close'].iloc[i]
        L4, L3, L2, L1, S1, S2, S3, S4 = calculate_camarilla(day_high, day_low, day_close)
        # These levels are valid for the next day
        start_idx = (i + 1) * 6  # 6 four-hour bars per day
        end_idx = min(start_idx + 6, n)
        if start_idx < n:
            L4_arr[start_idx:end_idx] = L4
            L3_arr[start_idx:end_idx] = L3
            L2_arr[start_idx:end_idx] = L2
            L1_arr[start_idx:end_idx] = L1
            S1_arr[start_idx:end_idx] = S1
            S2_arr[start_idx:end_idx] = S2
            S3_arr[start_idx:end_idx] = S3
            S4_arr[start_idx:end_idx] = S4
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(L4_arr[i]) or np.isnan(L3_arr[i]) or 
            np.isnan(L2_arr[i]) or np.isnan(L1_arr[i]) or np.isnan(S1_arr[i]) or 
            np.isnan(S2_arr[i]) or np.isnan(S3_arr[i]) or np.isnan(S4_arr[i]) or
            np.isnan(vol_ma.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma.iloc[i] * 1.5
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Long conditions: break above L3 with volume and uptrend
        if vol_ok and uptrend and close[i] > L3_arr[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short conditions: break below S3 with volume and downtrend
        elif vol_ok and downtrend and close[i] < S3_arr[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: reverse signal or volatility stop
        elif position == 1 and (close[i] < L1_arr[i] or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > S1_arr[i] or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals