#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Touch_Reversal_v1
Hypothesis: Uses Camarilla pivot levels from daily timeframe to identify potential reversal points.
In ranging markets (Choppiness > 61.8), we take long positions when price touches or crosses above S1/S2/S3 levels,
and short positions when price touches or crosses below R1/R2/R3 levels. In trending markets (Choppiness < 38.2),
we avoid trades to prevent whipsaw. Designed for low trade frequency by requiring both Camarilla touch and high Choppiness.
Works in both bull and bear markets by focusing on mean reversion in ranging conditions at key support/resistance levels.
"""

name = "12h_Camarilla_Pivot_Touch_Reversal_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- Camarilla Pivot Levels (based on previous day) ---
    # Formula based on previous day's OHLC
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Pivot point
    pivot = (high_prev + low_prev + close_prev) / 3.0
    
    # Camarilla levels
    r1 = close_prev + (high_prev - low_prev) * 1.1 / 12
    r2 = close_prev + (high_prev - low_prev) * 1.1 / 6
    r3 = close_prev + (high_prev - low_prev) * 1.1 / 4
    s1 = close_prev - (high_prev - low_prev) * 1.1 / 12
    s2 = close_prev - (high_prev - low_prev) * 1.1 / 6
    s3 = close_prev - (high_prev - low_prev) * 1.1 / 4
    
    # --- Choppiness Index (14-period) on 1d data ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true ranges over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index = 100 * log10(tr_sum / (atr * 14)) / log10(14)
    chop = 100 * np.log10(tr_sum / (atr * 14)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)  # Replace NaN with neutral value
    
    # Align Camarilla levels and Choppiness Index to 12h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Regime filters
        choppy_market = chop_aligned[i] > 61.8  # Ranging market
        trending_market = chop_aligned[i] < 38.2  # Trending market
        
        if position == 0:
            # Only trade in choppy/ranging markets
            if choppy_market:
                # Long when price touches or crosses above S1/S2/S3 (support levels)
                if (close[i] >= s1_aligned[i] or 
                    close[i] >= s2_aligned[i] or 
                    close[i] >= s3_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short when price touches or crosses below R1/R2/R3 (resistance levels)
                elif (close[i] <= r1_aligned[i] or 
                      close[i] <= r2_aligned[i] or 
                      close[i] <= r3_aligned[i]):
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions: price returns to pivot or regime changes
            if position == 1:
                # Exit long: price returns to pivot or market becomes trending
                exit_signal = (close[i] <= pivot[i]) or trending_market
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to pivot or market becomes trending
                exit_signal = (close[i] >= pivot[i]) or trending_market
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals