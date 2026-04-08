#!/usr/bin/env python3
# [24887] 6h_1d_weekly_pivot_v1
# Hypothesis: 6-hour price action relative to daily and weekly pivot levels. 
# Long when price pulls back to weekly pivot (or S1/S2) in an uptrend (price > weekly pivot).
# Short when price rallies to weekly pivot (or R1/R2) in a downtrend (price < weekly pivot).
# Uses weekly pivot as dynamic support/resistance and daily trend filter.
# Designed for low frequency (15-25 trades/year) to minimize fee drag in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_weekly_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for pivot calculation and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard floor trader's pivots)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pp_1d - low_1d
    s1_1d = 2 * pp_1d - high_1d
    r2_1d = pp_1d + (high_1d - low_1d)
    s2_1d = pp_1d - (high_1d - low_1d)
    
    # Calculate weekly pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pp_1w - low_1w
    s1_1w = 2 * pp_1w - high_1w
    r2_1w = pp_1w + (high_1w - low_1w)
    s2_1w = pp_1w - (high_1w - low_1w)
    
    # Align daily and weekly pivots to 6-hour timeframe
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Daily trend filter: price vs daily pivot
    trend_up = close > pp_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if pivot data not ready
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(pp_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly S1 or daily trend turns down
            if price < s1_1w_aligned[i] or not trend_up[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above weekly R1 or daily trend turns up
            if price > r1_1w_aligned[i] or trend_up[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price near weekly S1/S2 in uptrend (above daily pivot)
            if trend_up[i] and (abs(price - s1_1w_aligned[i]) / price < 0.005 or 
                               abs(price - s2_1w_aligned[i]) / price < 0.005):
                position = 1
                signals[i] = 0.25
            # Enter short: price near weekly R1/R2 in downtrend (below daily pivot)
            elif not trend_up[i] and (abs(price - r1_1w_aligned[i]) / price < 0.005 or 
                                     abs(price - r2_1w_aligned[i]) / price < 0.005):
                position = -1
                signals[i] = -0.25
    
    return signals