#!/usr/bin/env python3
# 1d_MultiTimeframe_Momentum_V1
# Hypothesis: Combine 1d momentum (4h EMA crossover) with 1w trend filter (1w EMA > 1w SMA) and volume confirmation.
# In bull markets: 4h EMA crosses above 4h EMA with 1w uptrend and volume surge = long.
# In bear markets: 4h EMA crosses below 4h EMA with 1w downtrend and volume surge = short.
# Uses 4h for entry timing and 1w for trend filter to reduce whipsaw. Target: 20-60 trades over 4 years.

name = "1d_MultiTimeframe_Momentum_V1"
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
    volume = prices['volume'].values
    
    # Get 4h data for entry signals
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA crossover (9 and 21)
    close_4h = df_4h['close'].values
    ema9_4h = pd.Series(close_4h).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 4h EMAs to 1d timeframe
    ema9_4h_aligned = align_htf_to_ltf(prices, df_4h, ema9_4h)
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) and SMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Align 1w trend filter to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # Volume confirmation: volume > 1.5 * 20-period average on 1d
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50)  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema9_4h_aligned[i]) or np.isnan(ema21_4h_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(sma50_1w_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1w: EMA50 > SMA50 = uptrend, EMA50 < SMA50 = downtrend
        is_uptrend = ema50_1w_aligned[i] > sma50_1w_aligned[i]
        is_downtrend = ema50_1w_aligned[i] < sma50_1w_aligned[i]
        
        if position == 0:
            # Long: 4h EMA9 crosses above EMA21 with uptrend on 1w and volume confirmation
            if (ema9_4h_aligned[i] > ema21_4h_aligned[i] and 
                ema9_4h_aligned[i-1] <= ema21_4h_aligned[i-1] and
                is_uptrend and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: 4h EMA9 crosses below EMA21 with downtrend on 1w and volume confirmation
            elif (ema9_4h_aligned[i] < ema21_4h_aligned[i] and 
                  ema9_4h_aligned[i-1] >= ema21_4h_aligned[i-1] and
                  is_downtrend and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if 4h EMA9 crosses below EMA21 or 1w trend turns down
            if (ema9_4h_aligned[i] < ema21_4h_aligned[i] or not is_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if 4h EMA9 crosses above EMA21 or 1w trend turns up
            if (ema9_4h_aligned[i] > ema21_4h_aligned[i] or not is_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals