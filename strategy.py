#!/usr/bin/env python3
"""
Hypothesis: 6-hour Choppiness Index regime filter with weekly pivot support/resistance and volume confirmation.
In choppy markets (CHOP > 61.8): mean-revert at weekly pivot S1/R1 levels.
In trending markets (CHOP < 38.2): breakout continuation at weekly pivot S2/R2 levels.
Weekly pivots provide institutional reference levels that work in both bull and bear markets.
Volume filter ensures participation. Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_chop(high, low, close, length=14):
    """Choppiness Index: measures whether market is choppy (high values) or trending (low values)"""
    if len(high) < length:
        return np.full_like(high, np.nan, dtype=np.float64)
    
    atr = np.zeros_like(high)
    for i in range(len(high)):
        if i == 0:
            atr[i] = high[i] - low[i]
        else:
            atr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Sum of ATR over period
    atr_sum = np.zeros_like(high)
    for i in range(length-1, len(high)):
        atr_sum[i] = np.sum(atr[i-length+1:i+1])
    
    # Highest high and lowest low over period
    highest_high = np.zeros_like(high)
    lowest_low = np.zeros_like(high)
    for i in range(length-1, len(high)):
        highest_high[i] = np.max(high[i-length+1:i+1])
        lowest_low[i] = np.min(low[i-length+1:i+1])
    
    # Chop calculation
    chop = np.full_like(high, np.nan, dtype=np.float64)
    for i in range(length-1, len(high)):
        if highest_high[i] != lowest_low[i]:
            log_val = np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(length)
            chop[i] = 100 * log_val
        else:
            chop[i] = 50.0  # neutral when no range
    
    return chop

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points: P, R1, S1, R2, S2, R3, S3"""
    pivot = (high + low + close) / 3
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
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and chop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly chop
    chop = calculate_chop(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 14)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Calculate weekly pivot points
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    pivot, r1, s1, r2, s2, r3, s3 = calculate_pivot_points(wk_high, wk_low, wk_close)
    
    # Align pivot levels
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Get daily volume for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need chop (14) + volume MA (20)
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 6h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Current levels
        chop_val = chop_aligned[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        
        # Volume filter: volume > 1.2x daily average
        vol_filter = vol_now > 1.2 * vol_ma
        
        if position == 0:
            # Choppy market (CHOP > 61.8): mean revert at S1/R1
            if chop_val > 61.8:
                if price_now <= s1_val and vol_filter:
                    signals[i] = size
                    position = 1
                elif price_now >= r1_val and vol_filter:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            # Trending market (CHOP < 38.2): breakout at S2/R2
            elif chop_val < 38.2:
                if price_now >= r2_val and vol_filter:
                    signals[i] = size
                    position = 1
                elif price_now <= s2_val and vol_filter:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            # Transition zone (38.2 <= CHOP <= 61.8): no trade
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: chop becomes too high (choppy) or price reaches opposite level
            if chop_val > 61.8 or price_now <= s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: chop becomes too high (choppy) or price reaches opposite level
            if chop_val > 61.8 or price_now >= r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ChopRegime_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0