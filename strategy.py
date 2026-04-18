#!/usr/bin/env python3
"""
1d_Weekly_Pivot_Point_Reversal_with_Trend_Filter
Hypothesis: Weekly pivot point reversals on 1d with trend filter using 1w EMA50.
Buy when price touches weekly S1 with bullish trend (close > EMA50) and sells at weekly R1.
Sell when price touches weekly R1 with bearish trend (close < EMA50) and buys back at weekly S1.
Designed for very low trade frequency (10-30/year) to minimize fee drag while capturing
mean-reversion moves in ranging markets and trend continuations in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for pivot point calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using previous week's data)
    # Pivot = (H + L + C) / 3
    # S1 = (2 * Pivot) - High
    # R1 = (2 * Pivot) - Low
    pivot = (high_1w + low_1w + close_1w) / 3
    s1 = (2 * pivot) - high_1w
    r1 = (2 * pivot) - low_1w
    
    # Align to 1d timeframe (wait for weekly bar to close)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 30)  # Warmup for EMA
    
    for i in range(start_idx, n):
        if (np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        s1_val = s1_aligned[i]
        r1_val = r1_aligned[i]
        ema50 = ema_50_aligned[i]
        
        if position == 0:
            # Long: price touches or goes below S1 with bullish trend
            if price <= s1_val and close[i] > ema50:
                signals[i] = 0.25
                position = 1
            # Short: price touches or goes above R1 with bearish trend
            elif price >= r1_val and close[i] < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price touches or goes above R1 (take profit)
            if price >= r1_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price touches or goes below S1 (take profit)
            if price <= s1_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Weekly_Pivot_Point_Reversal_with_Trend_Filter"
timeframe = "1d"
leverage = 1.0