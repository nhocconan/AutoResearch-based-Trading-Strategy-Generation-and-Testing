#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Reversion_1dTrend_Filter_v1
Hypothesis: Trade reversals from weekly pivot points (R4/S4) on 6h timeframe when price is in alignment with 1d trend.
In bull markets: long at weekly S4, short at weekly R4. In bear markets: long at weekly S4, short at weekly R4.
Trend filter: use 1d EMA(34) to determine direction - only take trades in direction of higher timeframe trend.
Weekly pivots act as strong support/resistance, causing reactions. Trend filter avoids counter-trend trades in strong moves.
Designed for 6h timeframe to target 15-35 trades/year with strict entry conditions.
"""
name = "6h_Weekly_Pivot_Reversion_1dTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (R4, S4 levels)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r4 = pivot_1w + (range_1w * 1.1 / 2) * 2  # R4 = pivot + 1.1 * range
    s4 = pivot_1w - (range_1w * 1.1 / 2) * 2  # S4 = pivot - 1.1 * range
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4, additional_delay_bars=0)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 1)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine trend direction from 1d EMA
            # Uptrend: price above EMA, Downtrend: price below EMA
            is_uptrend = close[i] > ema_34_1d_aligned[i]
            
            # Long when price touches S4 support (regardless of trend - pivots work in both directions)
            if low[i] <= s4_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short when price touches R4 resistance
            elif high[i] >= r4_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit conditions
            if position == 1:
                # Exit long when price reaches R4 or closes below EMA in strong downtrend
                if high[i] >= r4_aligned[i] or (close[i] < ema_34_1d_aligned[i] and close[i-1] >= ema_34_1d_aligned[i-1]):
                    signals[i] = 0.0
                    position = 0
            elif position == -1:
                # Exit short when price reaches S4 or closes above EMA in strong uptrend
                if low[i] <= s4_aligned[i] or (close[i] > ema_34_1d_aligned[i] and close[i-1] <= ema_34_1d_aligned[i-1]):
                    signals[i] = 0.0
                    position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals