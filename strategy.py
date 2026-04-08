#!/usr/bin/env python3
# 6h_1d_1w_pivots_momentum_v1
# Hypothesis: Combine weekly trend with daily pivot points and momentum confirmation.
# In weekly uptrend: go long when price crosses above daily pivot with momentum.
# In weekly downtrend: go short when price crosses below daily pivot with momentum.
# Uses momentum (ROC) to filter false breakouts and weekly EMA for trend filter.
# Designed for 6-15 trades/year (24-60 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_pivots_momentum_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA21 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot = np.zeros(len(high_1d))
    support1 = np.zeros(len(high_1d))
    resistance1 = np.zeros(len(high_1d))
    support2 = np.zeros(len(high_1d))
    resistance2 = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        # Previous day's OHLC
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        # Pivot point calculation
        pivot[i] = (prev_high + prev_low + prev_close) / 3.0
        support1[i] = (2 * pivot[i]) - prev_high
        resistance1[i] = (2 * pivot[i]) - prev_low
        support2[i] = pivot[i] - (prev_high - prev_low)
        resistance2[i] = pivot[i] + (prev_high - prev_low)
    
    # Align pivot points to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    support1_aligned = align_htf_to_ltf(prices, df_1d, support1)
    resistance1_aligned = align_htf_to_ltf(prices, df_1d, resistance1)
    support2_aligned = align_htf_to_ltf(prices, df_1d, support2)
    resistance2_aligned = align_htf_to_ltf(prices, df_1d, resistance2)
    
    # Momentum: Rate of Change (ROC) over 10 periods
    roc = np.zeros(n)
    for i in range(10, n):
        if close[i-10] != 0:
            roc[i] = (close[i] - close[i-10]) / close[i-10] * 100.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema21_1w_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(support1_aligned[i]) or np.isnan(resistance1_aligned[i]) or
            np.isnan(roc[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Momentum threshold: ROC > 0.5% for long, ROC < -0.5% for short
        mom_long = roc[i] > 0.5
        mom_short = roc[i] < -0.5
        
        if position == 1:  # Long position
            # Exit: price falls below support1 or weekly trend breaks
            if close[i] < support1_aligned[i] or close[i] < ema21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above resistance1 or weekly trend breaks
            if close[i] > resistance1_aligned[i] or close[i] > ema21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price crosses above pivot with momentum and weekly uptrend
            if (close[i] > pivot_aligned[i] and mom_long and 
                close[i-1] <= pivot_aligned[i-1] and  # Cross above pivot
                close[i] > ema21_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price crosses below pivot with momentum and weekly downtrend
            elif (close[i] < pivot_aligned[i] and mom_short and 
                  close[i-1] >= pivot_aligned[i-1] and  # Cross below pivot
                  close[i] < ema21_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals