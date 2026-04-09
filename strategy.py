#!/usr/bin/env python3
# 1d_1w_camarilla_pullback_v2
# Hypothesis: Daily strategy using weekly Camarilla pivot levels with pullback entries.
# Long: Price pulls back to weekly R3 level during uptrend (price > weekly close).
# Short: Price pulls back to weekly S3 level during downtrend (price < weekly close).
# Exit: Price reaches weekly pivot point or opposite S3/R3 level.
# Uses weekly Camarilla for key support/resistance, daily for execution with trend filter.
# Target: 7-25 trades/year (30-100 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pullback_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for Camarilla pivot levels (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    r3_1w = close_1w + range_1w * 1.1 / 4.0
    s3_1w = close_1w - range_1w * 1.1 / 4.0
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(close_1w_aligned[i]) or
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price reaches weekly pivot or breaks below S3
            if close[i] <= pivot_1w_aligned[i] or close[i] < s3_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price reaches weekly pivot or breaks above R3
            if close[i] >= pivot_1w_aligned[i] or close[i] > r3_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for pullback to Camarilla levels with trend filter
            bullish_setup = (close[i] >= r3_1w_aligned[i] * 0.995 and close[i] <= r3_1w_aligned[i] * 1.005) and (close[i] > close_1w_aligned[i])
            bearish_setup = (close[i] <= s3_1w_aligned[i] * 1.005 and close[i] >= s3_1w_aligned[i] * 0.995) and (close[i] < close_1w_aligned[i])
            
            if bullish_setup:
                position = 1
                signals[i] = 0.25
            elif bearish_setup:
                position = -1
                signals[i] = -0.25
    
    return signals