#!/usr/bin/env python3
"""
12h_1w_1d_Camarilla_Pivot_MeanReversion
Hypothesis: Combines weekly trend filter with daily Camarilla pivot mean reversion on 12h timeframe.
In bull markets (price > weekly EMA50), looks for longs at L3 and shorts at H3.
In bear markets (price < weekly EMA50), looks for longs at L4 and shorts at H4.
Uses volume confirmation (volume > 1.5x 20-period average) to avoid false breaks.
Designed for low turnover: targets 12-37 trades/year on 12h (50-150 total over 4 years).
Works in both bull and bear markets by adapting pivot levels to trend context.
"""

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
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    pivot = (high_prev + low_prev + close_prev) / 3
    range_prev = high_prev - low_prev
    
    # Camarilla levels
    H4 = pivot + (range_prev * 1.1 / 2)
    H3 = pivot + (range_prev * 1.1 / 4)
    L3 = pivot - (range_prev * 1.1 / 4)
    L4 = pivot - (range_prev * 1.1 / 2)
    
    # Align weekly trend to 12h
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Align daily Camarilla levels to 12h
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_confirmation = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(H4_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(L4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine market regime: bull if price > weekly EMA50, bear otherwise
        is_bull = close[i] > ema50_1w_aligned[i]
        
        # Select pivot levels based on regime
        if is_bull:
            # Bull market: look for longs at L3, shorts at H3
            long_level = L3_aligned[i]
            short_level = H3_aligned[i]
        else:
            # Bear market: look for longs at L4, shorts at H4
            long_level = L4_aligned[i]
            short_level = H4_aligned[i]
        
        # Check for entry signals with volume confirmation
        long_signal = (close[i] <= long_level) and volume_confirmation[i]
        short_signal = (close[i] >= short_level) and volume_confirmation[i]
        
        # Exit conditions: price reaches opposite level or reverses
        exit_long = (close[i] >= short_level) or (not volume_confirmation[i] and position == 1)
        exit_short = (close[i] <= long_level) or (not volume_confirmation[i] and position == -1)
        
        # Update position and signal
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_1w_1d_Camarilla_Pivot_MeanReversion"
timeframe = "12h"
leverage = 1.0