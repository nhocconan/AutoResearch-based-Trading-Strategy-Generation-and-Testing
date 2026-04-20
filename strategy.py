#!/usr/bin/env python3
"""
1d_1w_Stochastic_Trend_v1
Concept: Weekly Stochastic(14,3,3) as trend filter with daily price action for entry.
- Long: %K > 50 and %D > 50 (bullish weekly trend) AND daily close > daily open (bullish daily candle)
- Short: %K < 50 and %D < 50 (bearish weekly trend) AND daily close < daily open (bearish daily candle)
- Exit: Weekly trend reversal (%K crosses below/above 50)
- Position sizing: 0.25
- Target: 10-25 trades/year (40-100 total over 4 years)
- Works in bull/bear: Weekly Stochastic defines primary trend, daily price action provides precise entry/exit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Stochastic_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for Stochastic calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # === Weekly: Stochastic(14,3,3) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate %K (fast stochastic)
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    k_fast = 100 * (close_1w - lowest_low) / (highest_high - lowest_low)
    # Handle division by zero
    k_fast = np.where((highest_high - lowest_low) == 0, 50, k_fast)
    
    # Calculate %D (3-period SMA of %K)
    d_slow = pd.Series(k_fast).rolling(window=3, min_periods=3).mean().values
    
    # Align weekly indicators to daily timeframe
    k_aligned = align_htf_to_ltf(prices, df_1w, k_fast)
    d_aligned = align_htf_to_ltf(prices, df_1w, d_slow)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # Ensure enough data for Stochastic
    
    for i in range(start_idx, n):
        # Get values
        k_val = k_aligned[i]
        d_val = d_aligned[i]
        
        # Skip if any value is NaN
        if np.isnan(k_val) or np.isnan(d_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Weekly bullish trend (%K>50 and %D>50) AND daily bullish candle
            if k_val > 50 and d_val > 50 and prices['close'].iloc[i] > prices['open'].iloc[i]:
                signals[i] = 0.25
                position = 1
            # Short: Weekly bearish trend (%K<50 and %D<50) AND daily bearish candle
            elif k_val < 50 and d_val < 50 and prices['close'].iloc[i] < prices['open'].iloc[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Weekly trend turns bearish (%K crosses below 50)
            if k_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Weekly trend turns bullish (%K crosses above 50)
            if k_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals