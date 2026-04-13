#!/usr/bin/env python3
"""
4h_1d_1w_Camarilla_Volume_Trend
Hypothesis: Uses Camarilla pivot levels from daily timeframe for breakout trading. 
Trades breakouts above R3 or below S3 with volume confirmation and weekly trend filter.
In trending markets, breaks through S3/R3 indicate strong momentum. 
Weekly trend filter ensures we only trade in direction of weekly momentum.
Volume expansion confirms institutional interest. 
Designed for 4h timeframe with target of 25-50 trades/year (100-200 total over 4 years).
Works in both bull and bear markets by following institutional breakouts with trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close arrays."""
    range_val = high - low
    # Handle zero range to avoid division by zero
    range_val = np.where(range_val == 0, 1e-10, range_val)
    
    C = close
    H = high
    L = low
    
    # Camarilla levels
    R1 = C + ((H - L) * 1.0833)
    R2 = C + ((H - L) * 1.1666)
    R3 = C + ((H - L) * 1.2500)
    R4 = C + ((H - L) * 1.5000)
    S1 = C - ((H - L) * 1.0833)
    S2 = C - ((H - L) * 1.1666)
    S3 = C - ((H - L) * 1.2500)
    S4 = C - ((H - L) * 1.5000)
    
    return R1, R2, R3, R4, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels on daily
    R1_1d, R2_1d, R3_1d, R4_1d, S1_1d, S2_1d, S3_1d, S4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly trend: bullish if weekly close > weekly open, bearish if weekly close < weekly open
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open
    
    # Align all data to 4h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    R2_1d_aligned = align_htf_to_ltf(prices, df_1d, R2_1d)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    S2_1d_aligned = align_htf_to_ltf(prices, df_1d, S2_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions with volume expansion and weekly trend filter
        
        # Long setup: price breaks above R3 with volume expansion, weekly bullish
        long_condition = (high[i] > R3_1d_aligned[i]) and volume_expansion[i] and weekly_bullish_aligned[i] > 0.5
        
        # Short setup: price breaks below S3 with volume expansion, weekly bearish
        short_condition = (low[i] < S3_1d_aligned[i]) and volume_expansion[i] and weekly_bullish_aligned[i] < 0.5
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif long_condition and position == 1:
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        elif short_condition and position == -1:
            signals[i] = -position_size
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_1d_1w_Camarilla_Volume_Trend"
timeframe = "4h"
leverage = 1.0