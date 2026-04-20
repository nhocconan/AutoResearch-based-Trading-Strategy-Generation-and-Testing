#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1-day pivot reversal zones
# Williams %R identifies overbought/oversold conditions
# 1-day pivot levels (S1, S2, R1, R2) act as support/resistance zones
# Strategy: Buy when Williams %R oversold (< -80) and price near S1/S2 pivot support
#           Sell when Williams %R overbought (> -20) and price near R1/R2 pivot resistance
# Works in both bull and bear markets by fading extremes at key pivot levels
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1-day data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate classic pivot points: P = (H+L+C)/3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Support and resistance levels
    s1 = 2 * pivot - high_1d
    s2 = pivot - (high_1d - low_1d)
    r1 = 2 * pivot - low_1d
    r2 = pivot + (high_1d - low_1d)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    
    # Calculate Williams %R on 6h timeframe (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    wr = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if NaN in indicators
        if np.isnan(wr[i]) or np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Williams %R oversold and price near S1 or S2 support
            long_condition = (wr[i] < -80) and (abs(price - s1_aligned[i]) / s1_aligned[i] < 0.02 or abs(price - s2_aligned[i]) / s2_aligned[i] < 0.02)
            
            # Short: Williams %R overbought and price near R1 or R2 resistance
            short_condition = (wr[i] > -20) and (abs(price - r1_aligned[i]) / r1_aligned[i] < 0.02 or abs(price - r2_aligned[i]) / r2_aligned[i] < 0.02)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns from oversold or price breaks below S2
            exit_condition = (wr[i] > -50) or (price < s2_aligned[i])
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns from overbought or price breaks above R2
            exit_condition = (wr[i] < -50) or (price > r2_aligned[i])
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1d_PivotReversal"
timeframe = "6h"
leverage = 1.0