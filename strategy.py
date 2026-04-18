#!/usr/bin/env python3
"""
1d_Weekly_Pivot_R1S1_Breakout_Trend_Filter_v1
Hypothesis: Weekly pivot R1/S1 level breakouts with daily trend filter (EMA50) capture momentum in both bull and bear markets.
Weekly pivots provide strong support/resistance. Daily EMA50 filters trend direction. Designed for low trade frequency (<10/year) to minimize fee drag on 1d timeframe.
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
    
    # Get weekly data for pivot calculation (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high']
    low_weekly = df_weekly['low']
    close_weekly = df_weekly['close']
    
    # Calculate weekly pivot points and R1/S1 levels
    pivot = (high_weekly + low_weekly + close_weekly) / 3
    r1 = 2 * pivot - low_weekly
    s1 = 2 * pivot - high_weekly
    
    # Align weekly pivot levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # Daily EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_50[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        ema_trend = ema_50[i]
        
        if position == 0:
            # Long: break above R1 with daily uptrend
            if price > r1_level and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with daily downtrend
            elif price < s1_level and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to S1 or breaks below daily EMA
            if price < s1_level or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to R1 or breaks above daily EMA
            if price > r1_level or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Weekly_Pivot_R1S1_Breakout_Trend_Filter_v1"
timeframe = "1d"
leverage = 1.0