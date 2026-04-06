#!/usr/bin/env python3
"""
6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation
Hypothesis: Donchian breakouts capture trend continuation, weekly pivots provide
institutional support/resistance, and volume filters false breakouts.
Works in bull markets (breakouts above weekly pivot) and bear markets
(breakdowns below weekly pivot). Target: 100-180 total trades over 4 years (25-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14447_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot points (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    pivot_w = (high_w + low_w + close_w) / 3
    # Support/resistance levels
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    r3_w = high_w + 2 * (pivot_w - low_w)
    s3_w = low_w - 2 * (high_w - pivot_w)
    
    # Align weekly pivot data to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_weekly, pivot_w)
    r3_w_aligned = align_htf_to_ltf(prices, df_weekly, r3_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_weekly, s3_w)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: avoid low volume periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (0.8 * vol_ma)
    
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
    start = max(20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(pivot_w_aligned[i]) or np.isnan(r3_w_aligned[i]) or
            np.isnan(s3_w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR stoploss
            if (close[i] <= donchian_low[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR stoploss
            if (close[i] >= donchian_high[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly pivot filter + volume
            long_breakout = close[i] > donchian_high[i]
            short_breakout = close[i] < donchian_low[i]
            
            # Weekly pivot filter: only long above R3, short below S3
            long_filter = close[i] > r3_w_aligned[i]
            short_filter = close[i] < s3_w_aligned[i]
            
            if long_breakout and long_filter and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and short_filter and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</lyzard>