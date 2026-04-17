#!/usr/bin/env python3
"""
Hypothesis: 6h 1-week and 1-day pivot points (R1/S1) act as support/resistance zones.
In ranging markets (CHOP > 61.8), price tends to revert from these levels.
In trending markets (CHOP < 38.2), breakouts beyond R1/S1 with volume continuation
signal momentum entries. Uses 1-week pivot for bias, 1-day pivot for entry/exit,
and CHOP regime filter to adapt strategy. Designed for low trade frequency (12-37/year)
to minimize fee drag in 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_chop(high, low, close, window=14):
    """Choppiness Index: higher = ranging, lower = trending"""
    atr_sum = pd.Series(np.maximum(np.maximum(high - low, 
                                         np.abs(high - np.roll(close, 1))),
                                   np.abs(low - np.roll(close, 1)))).rolling(window, min_periods=window).sum()
    highest_high = pd.Series(high).rolling(window, min_periods=window).max()
    lowest_low = pd.Series(low).rolling(window, min_periods=window).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(window)
    return chop.fillna(50).values  # neutral when undefined

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points: P, R1, S1, R2, S2, R3, S3"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, s1, r2, s2, r3, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1-week data for bias and pivot context ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Weekly pivot points (for bias)
    _, wp_r1, wp_s1, _, _, _, _ = calculate_pivot_points(
        df_1w['high'].values, 
        df_1w['low'].values, 
        df_1w['close'].values
    )
    # Weekly bias: price above weekly R1 = bullish bias, below S1 = bearish bias
    weekly_bias = np.where(close[:, None] > wp_r1, 1, 
                          np.where(close[:, None] < wp_s1, -1, 0))
    weekly_bias = weekly_bias[:, 0]  # take first column (should be same for all rows in window)
    
    # Align weekly bias to 6h
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    
    # === 1-day data for entry/exit levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Daily pivot points (for entry/exit)
    dp_pivot, dp_r1, dp_s1, dp_r2, dp_s2, _, _ = calculate_pivot_points(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values
    )
    
    # Align daily pivot levels to 6h
    dp_r1_aligned = align_htf_to_ltf(prices, df_1d, dp_r1)
    dp_s1_aligned = align_htf_to_ltf(prices, df_1d, dp_s1)
    dp_r2_aligned = align_htf_to_ltf(prices, df_1d, dp_r2)
    dp_s2_aligned = align_htf_to_ltf(prices, df_1d, dp_s2)
    
    # === 6h indicators ===
    # Chop regime filter (14-period)
    chop = calculate_chop(high, low, close, window=14)
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Initialize signals
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    # Start after warmup period
    start_idx = max(20, 14)  # volume MA and chop lookback
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(chop[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(dp_r1_aligned[i]) or np.isnan(dp_s1_aligned[i]) or
            np.isnan(weekly_bias_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20[i]
        bias = weekly_bias_aligned[i]
        
        # Regime determination
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        if position == 0:
            # === ENTRY LOGIC ===
            if is_ranging:
                # Mean reversion at daily S1/R1 with weekly bias filter
                # Long: price at/d below S1 with bullish or neutral weekly bias
                if price <= dp_s1_aligned[i] * 1.001 and bias >= 0:  # 0.1% buffer
                    if vol > 1.5 * vol_ma:  # volume confirmation
                        signals[i] = 0.25
                        position = 1
                # Short: price at/above R1 with bearish or neutral weekly bias
                elif price >= dp_r1_aligned[i] * 0.999 and bias <= 0:
                    if vol > 1.5 * vol_ma:
                        signals[i] = -0.25
                        position = -1
            else:  # trending or transitional
                # Breakout continuation with weekly bias alignment
                # Long: break above R1 with bullish bias
                if price > dp_r1_aligned[i] and bias > 0:
                    if vol > 2.0 * vol_ma:  # stronger volume for breakout
                        signals[i] = 0.25
                        position = 1
                # Short: break below S1 with bearish bias
                elif price < dp_s1_aligned[i] and bias < 0:
                    if vol > 2.0 * vol_ma:
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:
            # === EXIT LONG ===
            # Take profit at daily R2 or stop loss if weekly bias turns bearish
            if price >= dp_r2_aligned[i] or bias < 0:
                signals[i] = 0.0
                position = 0
            # Stop loss if price breaks below S1 (failed mean reversion/breakout)
            elif price < dp_s1_aligned[i] * 0.995:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # === EXIT SHORT ===
            # Take profit at daily S2 or stop loss if weekly bias turns bullish
            if price <= dp_s2_aligned[i] or bias > 0:
                signals[i] = 0.0
                position = 0
            # Stop loss if price breaks above R1 (failed mean reversion/breakout)
            elif price > dp_r1_aligned[i] * 1.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1W_Pivot_Bias_1D_Pivot_Entry_CHOP"
timeframe = "6h"
leverage = 1.0