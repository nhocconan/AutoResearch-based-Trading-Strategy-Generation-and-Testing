#!/usr/bin/env python3
"""
6h Weekly Pivot + Volume Confirmation
Hypothesis: Weekly pivot levels act as strong support/resistance. Price reacting at these levels with volume confirmation provides high-probability entries. Works in both bull and bear markets by fading extremes (R3/S3) and continuing breaks (R4/S4).
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14347_6h_weekly_pivot_vol_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    pivot_w = (high_w + low_w + close_w) / 3.0
    # Calculate support/resistance levels
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    r3_w = high_w + 2 * (pivot_w - low_w)
    s3_w = low_w - 2 * (high_w - pivot_w)
    r4_w = r3_w + (high_w - low_w)
    s4_w = s3_w - (high_w - low_w)
    
    # Align weekly levels to 6h timeframe
    pivot_w_a = align_htf_to_ltf(prices, df_w, pivot_w)
    r1_w_a = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_a = align_htf_to_ltf(prices, df_w, s1_w)
    r2_w_a = align_htf_to_ltf(prices, df_w, r2_w)
    s2_w_a = align_htf_to_ltf(prices, df_w, s2_w)
    r3_w_a = align_htf_to_ltf(prices, df_w, r3_w)
    s3_w_a = align_htf_to_ltf(prices, df_w, s3_w)
    r4_w_a = align_htf_to_ltf(prices, df_w, r4_w)
    s4_w_a = align_htf_to_ltf(prices, df_w, s4_w)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: avoid low volume periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.2 * vol_ma)  # Require above average volume
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(pivot_w_a[i]) or np.isnan(r3_w_a[i]) or np.isnan(s3_w_a[i]) or \
           np.isnan(r4_w_a[i]) or np.isnan(s4_w_a[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: stoploss or reversal signal
            if close[i] <= entry_price - 2.5 * atr[i] or \
               (close[i] < pivot_w_a[i] and low[i] <= s1_w_a[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: stoploss or reversal signal
            if close[i] >= entry_price + 2.5 * atr[i] or \
               (close[i] > pivot_w_a[i] and high[i] >= r1_w_a[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: fade at R3/S3, break at R4/S4 with volume
            # Long: price touches/falls below S3 with reversal + volume
            long_setup = (low[i] <= s3_w_a[i] * 1.001) and (close[i] > s3_w_a[i]) and vol_filter[i]
            # Short: price touches/rises above R3 with rejection + volume
            short_setup = (high[i] >= r3_w_a[i] * 0.999) and (close[i] < r3_w_a[i]) and vol_filter[i]
            # Long breakout: price breaks above R4 with volume
            long_breakout = (high[i] >= r4_w_a[i] * 0.999) and (close[i] > r4_w_a[i]) and vol_filter[i]
            # Short breakdown: price breaks below S4 with volume
            short_breakout = (low[i] <= s4_w_a[i] * 1.001) and (close[i] < s4_w_a[i]) and vol_filter[i]
            
            if long_setup or long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup or short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals