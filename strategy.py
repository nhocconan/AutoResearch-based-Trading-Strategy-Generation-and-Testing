#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotFilter_v1
Hypothesis: 6h Donchian(20) breakout filtered by weekly pivot direction.
In weekly bullish bias (price above weekly pivot): only long breakouts above Donchian upper.
In weekly bearish bias (price below weekly pivot): only short breakouts below Donchian lower.
Weekly pivot calculated from prior weekly OHLC. Uses ATR(14) stoploss (2.0x) and discrete sizing (0.25).
Designed to capture strong momentum moves aligned with weekly structure, reducing false breakouts.
Timeframe: 6h, uses 1w HTF for weekly pivot.
Target: 50-150 total trades over 4 years = 12-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for weekly pivot)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === 1w OHLC for weekly pivot calculation (based on previous weekly bar) ===
    df_1w_open = df_1w['open'].values
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    
    # Calculate weekly pivot and support/resistance levels
    pivot_1w = (df_1w_high + df_1w_low + df_1w_close) / 3.0
    r1_1w = 2.0 * pivot_1w - df_1w_low
    s1_1w = 2.0 * pivot_1w - df_1w_high
    r2_1w = pivot_1w + (df_1w_high - df_1w_low)
    s2_1w = pivot_1w - (df_1w_high - df_1w_low)
    r3_1w = df_1w_high + 2.0 * (pivot_1w - df_1w_low)
    s3_1w = df_1w_low - 2.0 * (df_1w_high - pivot_1w)
    
    # Align 1w pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # === 6h Donchian(20) channels ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) 
            or np.isnan(pivot_1w_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        pivot = pivot_1w_aligned[i]
        
        if position == 0:
            # Weekly bias-based entries
            if price > pivot:  # Weekly bullish bias
                # Only look for long breakouts
                if price > upper:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
            elif price < pivot:  # Weekly bearish bias
                # Only look for short breakouts
                if price < lower:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR) or break below weekly pivot
            if price < entry_price - 2.0 * atr[i] or price < pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR) or break above weekly pivot
            if price > entry_price + 2.0 * atr[i] or price > pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotFilter_v1"
timeframe = "6h"
leverage = 1.0