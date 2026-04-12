#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_1w_pivots_reversion_v1
# Uses weekly pivot points (PP) and 1-day Bollinger Bands to identify reversion zones.
# Long when price crosses below weekly S1 AND closes below 1d BB lower band (oversold in weekly downtrend).
# Short when price crosses above weekly R1 AND closes above 1d BB upper band (overbought in weekly uptrend).
# Exits when price returns to weekly pivot point.
# Works in both bull and bear markets by fading extremes relative to weekly structure.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_1d_1w_pivots_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (based on previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point and support/resistance levels
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r1_1w = pp_1w + range_1w  # Weekly R1
    s1_1w = pp_1w - range_1w  # Weekly S1
    
    # Align weekly levels to 6h timeframe
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Get daily data for Bollinger Bands (20-period, 2 std)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 20-period SMA and standard deviation
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_lower = sma_20 - 2 * std_20
    bb_upper = sma_20 + 2 * std_20
    
    # Align Bollinger Bands to 6h timeframe
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(bb_upper_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: price crosses below weekly S1 AND closes below BB lower band
        long_signal = (close[i] < s1_1w_aligned[i] and 
                      close[i] < bb_lower_aligned[i] and
                      (i == 50 or close[i-1] >= s1_1w_aligned[i-1]))
        
        # Short signal: price crosses above weekly R1 AND closes above BB upper band
        short_signal = (close[i] > r1_1w_aligned[i] and 
                       close[i] > bb_upper_aligned[i] and
                       (i == 50 or close[i-1] <= r1_1w_aligned[i-1]))
        
        # Exit when price returns to weekly pivot point
        exit_long = position == 1 and close[i] >= pp_1w_aligned[i]
        exit_short = position == -1 and close[i] <= pp_1w_aligned[i]
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long:
            position = 0
            signals[i] = 0.0
        elif exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals