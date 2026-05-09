#!/usr/bin/env python3
# 2025-06-22 | 1d_1W_Pivot_Trend_Filtered
# Hypothesis: Weekly pivot levels (weekly high/low) act as strong support/resistance on daily timeframe.
# Long when daily close breaks above weekly high with price above weekly EMA200 (uptrend filter).
# Short when daily close breaks below weekly low with price below weekly EMA200 (downtrend filter).
# Weekly EMA200 provides robust trend filter that adapts to bull/bear markets.
# Designed for very low trade frequency (<25/year) to minimize fee drag in ranging/bear markets.

name = "1d_1W_Pivot_Trend_Filtered"
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
    
    # Get weekly data for pivot levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's values for pivot calculation (weekly high/low)
    ph_1w = np.concatenate([[high_1w[0]], high_1w[:-1]])  # previous weekly high
    pl_1w = np.concatenate([[low_1w[0]], low_1w[:-1]])   # previous weekly low
    
    # Weekly pivot levels: use previous week's high/low as support/resistance
    weekly_high = ph_1w
    weekly_low = pl_1w
    
    # Align weekly pivot levels to daily timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Calculate weekly EMA200 for trend filter
    ema_200_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 200:
        # Initialize EMA200 with SMA of first 200 values
        ema_200_1w[199] = np.mean(close_1w[0:200])
        for i in range(200, len(close_1w)):
            ema_200_1w[i] = (ema_200_1w[i-1] * 199 + close_1w[i]) / 200
    
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure weekly EMA200 is ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly high AND uptrend (price > weekly EMA200)
            if close[i] > weekly_high_aligned[i] and close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly low AND downtrend (price < weekly EMA200)
            elif close[i] < weekly_low_aligned[i] and close[i] < ema_200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below weekly low OR trend reversal (price < weekly EMA200)
            if close[i] < weekly_low_aligned[i] or close[i] < ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly high OR trend reversal (price > weekly EMA200)
            if close[i] > weekly_high_aligned[i] or close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals