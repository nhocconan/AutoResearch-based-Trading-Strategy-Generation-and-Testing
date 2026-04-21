#!/usr/bin/env python3
"""
6h_RangeReversal_FibPivots_1dTrendFilter_V1
Hypothesis: In ranging markets, price reverses at Fibonacci-based pivot levels (R1/S1) from the prior 1d session. Trades are filtered by 1d trend (price above/below 50 EMA) to avoid counter-trend entries. Works in bull/bear by only taking reversal trades aligned with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_fib_pivots(high, low, close):
    """Calculate Fibonacci pivot points: R1, S1, pivot"""
    pivot = (high + low + close) / 3
    range_val = high - low
    r1 = pivot + 0.382 * range_val
    s1 = pivot - 0.382 * range_val
    return pivot, r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Load 1d data once for pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Fibonacci pivots on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = np.full_like(close_1d, np.nan)
    r1_1d = np.full_like(close_1d, np.nan)
    s1_1d = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        pivot_1d[i], r1_1d[i], s1_1d[i] = calculate_fib_pivots(high_1d[i], low_1d[i], close_1d[i])
    
    # Calculate 50 EMA for trend filter on daily data
    close_series = pd.Series(close_1d)
    ema_50_1d = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 1d indicators to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h price data
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        pivot = pivot_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        ema_50 = ema_50_1d_aligned[i]
        
        # Determine 1d trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_trend = price > ema_50
        bearish_trend = price < ema_50
        
        if position == 0:
            # Long: price at or below S1 in bullish 1d trend
            if bullish_trend and price <= s1:
                signals[i] = 0.25
                position = 1
            # Short: price at or above R1 in bearish 1d trend
            elif bearish_trend and price >= r1:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches pivot or trend turns bearish
            if price >= pivot or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches pivot or trend turns bullish
            if price <= pivot or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RangeReversal_FibPivots_1dTrendFilter_V1"
timeframe = "6h"
leverage = 1.0