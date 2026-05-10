#!/usr/bin/env python3
# 1d_Supertrend_1wTrend_Filter
# Hypothesis: Use Supertrend on daily timeframe with weekly trend filter to capture major trends.
# In strong trends (weekly close above/below EMA200), follow Supertrend signals; in weak trends, stay flat.
# Designed for low frequency (10-25 trades/year) with high conviction trades. Works in bull via upward Supertrend
# and bear via downward Supertrend, filtered by weekly trend to avoid counter-trend whipsaws.

name = "1d_Supertrend_1wTrend_Filter"
timeframe = "1d"
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
    
    # Daily data for Supertrend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR
    tr1 = pd.Series(df_1d['high']) - pd.Series(df_1d['low'])
    tr2 = pd.Series(df_1d['high']).abs() - pd.Series(df_1d['close']).shift(1).abs()
    tr3 = pd.Series(df_1d['low']).abs() - pd.Series(df_1d['close']).shift(1).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()
    
    # Calculate upper and lower bands
    hl_avg = (pd.Series(df_1d['high']) + pd.Series(df_1d['low'])) / 2
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.full(len(df_1d), np.nan)
    direction = np.full(len(df_1d), 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(atr_period, len(df_1d)):
        if i == atr_period:
            supertrend[i] = upper_band.iloc[i]
            direction[i] = -1  # start in downtrend waiting for breakout
        else:
            if supertrend[i-1] == upper_band.iloc[i-1]:
                if close[i] <= upper_band.iloc[i]:
                    supertrend[i] = upper_band.iloc[i]
                    direction[i] = -1
                else:
                    supertrend[i] = lower_band.iloc[i]
                    direction[i] = 1
            else:
                if close[i] >= lower_band.iloc[i]:
                    supertrend[i] = lower_band.iloc[i]
                    direction[i] = 1
                else:
                    supertrend[i] = upper_band.iloc[i]
                    direction[i] = -1
    
    # Weekly trend filter: price relative to EMA200
    close_1w = pd.Series(df_1w['close'])
    ema200_1w = close_1w.ewm(span=200, adjust=False, min_periods=200).mean()
    weekly_uptrend = close_1w > ema200_1w
    weekly_downtrend = close_1w < ema200_1w
    
    # Align to daily timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Supertrend flips to uptrend AND weekly uptrend
            if direction_aligned[i] == 1 and weekly_uptrend_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Enter short: Supertrend flips to downtrend AND weekly downtrend
            elif direction_aligned[i] == -1 and weekly_downtrend_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when Supertrend flips to downtrend or weekly trend fails
            if direction_aligned[i] == -1 or weekly_uptrend_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when Supertrend flips to uptrend or weekly trend fails
            if direction_aligned[i] == 1 or weekly_downtrend_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals