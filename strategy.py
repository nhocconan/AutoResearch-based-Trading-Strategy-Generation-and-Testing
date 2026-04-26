#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Donchian_Breakout_1dTrendFilter
Hypothesis: Combines weekly Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) with daily trend filter and 6h Donchian(20) breakout confirmation. In ranging markets (price between R3/S3), fade extremes. In trending markets (price outside R4/S4), breakout continuation. Uses 1d EMA50 for trend regime detection. Designed for low trade frequency (~10-25/year) with discrete sizing 0.25 to minimize fee drag. Works in both bull and bear via adaptive regime logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load weekly data ONCE before loop for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    # Typical Price = (H + L + C) / 3
    tp_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    # Camarilla: R4 = TP + 1.1*(H-L)/2, R3 = TP + 1.1*(H-L)/4, S3 = TP - 1.1*(H-L)/4, S4 = TP - 1.1*(H-L)/2
    tp_1w_vals = tp_1w.values
    weekly_range = df_1w['high'].values - df_1w['low'].values
    r4_1w = tp_1w_vals + 1.1 * weekly_range / 2.0
    r3_1w = tp_1w_vals + 1.1 * weekly_range / 4.0
    s3_1w = tp_1w_vals - 1.1 * weekly_range / 4.0
    s4_1w = tp_1w_vals - 1.1 * weekly_range / 2.0
    
    # Align weekly pivot levels to 6h timeframe (completed weekly bar only)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h Donchian(20) for breakout confirmation
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for 1d EMA, 20 for Donchian
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r4_1w_aligned[i]) or
            np.isnan(r3_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i]) or
            np.isnan(s4_1w_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r4 = r4_1w_aligned[i]
        r3 = r3_1w_aligned[i]
        s3 = s3_1w_aligned[i]
        s4 = s4_1w_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        upper_break = donchian_high[i]
        lower_break = donchian_low[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry based on weekly pivot zones and trend
            # Regime 1: Ranging market (between R3 and S3) -> mean reversion at extremes
            # Regime 2: Trending market (above R4 or below S4) -> breakout continuation
            # Regime 3: Transition zone (between S3/R3 and S4/R4) -> wait for clear signal
            
            in_range = s3 < close_val < r3  # Inside R3-S3: mean reversion zone
            above_r4 = close_val > r4       # Above R4: bullish breakout zone
            below_s4 = close_val < s4       # Below S4: bearish breakout zone
            
            # Long entry conditions:
            # 1. In range, price at S3 support (mean reversion long)
            # 2. Above R4, breaking Donchian high (bullish continuation)
            long_mean_rev = in_range and close_val <= s3 * 1.001  # Near S3 support
            long_breakout = above_r4 and close_val > upper_break  # Break above Donchian high
            
            # Short entry conditions:
            # 1. In range, price at R3 resistance (mean reversion short)
            # 2. Below S4, breaking Donchian low (bearish continuation)
            short_mean_rev = in_range and close_val >= r3 * 0.999  # Near R3 resistance
            short_breakout = below_s4 and close_val < lower_break  # Break below Donchian low
            
            # Filter by 1d trend: only take mean reversion trades against trend,
            # only take breakout trades with trend
            if (long_mean_rev and close_val < ema_50_val) or (long_breakout and close_val > ema_50_val):
                signals[i] = size
                position = 1
                entry_price = close_val
            elif (short_mean_rev and close_val > ema_50_val) or (short_breakout and close_val < ema_50_val):
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit conditions
            # 1. Price reaches R3 (take profit at resistance in ranging market)
            # 2. Price breaks below S4 (stop loss if trend fails)
            # 3. Donchian low break (trailing stop)
            if close_val >= r3 * 0.999 or close_val < s4 or close_val < lower_break:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit conditions
            # 1. Price reaches S3 (take profit at support in ranging market)
            # 2. Price breaks above R4 (stop loss if trend fails)
            # 3. Donchian high break (trailing stop)
            if close_val <= s3 * 1.001 or close_val > r4 or close_val > upper_break:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Weekly_Pivot_Donchian_Breakout_1dTrendFilter"
timeframe = "6h"
leverage = 1.0