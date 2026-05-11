#!/usr/bin/env python3
name = "6h_LongTermSupportResistance_Breakout_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for long-term support/resistance and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 50-period high and low on weekly data (long-term SR levels)
    high_50w = pd.Series(high_1w).rolling(window=50, min_periods=50).max().values
    low_50w = pd.Series(low_1w).rolling(window=50, min_periods=50).min().values
    
    # Calculate 20-period EMA on weekly data for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly levels and trend to 6h timeframe
    high_50w_aligned = align_htf_to_ltf(prices, df_1w, high_50w)
    low_50w_aligned = align_htf_to_ltf(prices, df_1w, low_50w)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation: current volume > 2.0x 50-period average
    vol_ma50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > 2.0 * vol_ma50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for weekly calculations
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(high_50w_aligned[i]) or np.isnan(low_50w_aligned[i]) or
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above 50-week high + weekly uptrend + volume spike
            if (close[i] > high_50w_aligned[i] and 
                close[i] > ema20_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below 50-week low + weekly downtrend + volume spike
            elif (close[i] < low_50w_aligned[i] and 
                  close[i] < ema20_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below 50-week EMA or breaks below 50-week low
            if (close[i] < ema20_1w_aligned[i] or 
                close[i] < low_50w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above 50-week EMA or breaks above 50-week high
            if (close[i] > ema20_1w_aligned[i] or 
                close[i] > high_50w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals