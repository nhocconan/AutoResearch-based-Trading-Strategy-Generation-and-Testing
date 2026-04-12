#!/usr/bin/env python3
"""
6h_1w_1d_Weekly_Pivot_Trend_Follow
Hypothesis: On 6h timeframe, follow weekly trend using weekly pivot points (PP) and trade in direction 
of weekly trend on breakouts of daily high/low with volume confirmation. Uses weekly pivot for 
trend direction (price > PP = uptrend, price < PP = downtrend) and daily range breakout for entry 
timing. Works in bull (buy breakouts above weekly PP) and bear (sell breakdowns below weekly PP). 
Target: 15-30 trades per year per symbol (60-120 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_Weekly_Pivot_Trend_Follow"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY PIVOT: Calculate from previous week's OHLC ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points using previous week's data
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    
    # Weekly Pivot Point (PP) = (High + Low + Close) / 3
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3
    # Weekly Support 1 (S1) = (2 * PP) - High
    weekly_s1 = (2 * weekly_pp) - weekly_high
    # Weekly Resistance 1 (R1) = (2 * PP) - Low
    weekly_r1 = (2 * weekly_pp) - weekly_low
    
    # Align weekly levels to 6h timeframe (already shifted for completed week)
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # === DAILY HIGH/LOW: For breakout entry ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Align daily high/low to 6h timeframe
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    
    # === VOLUME FILTER: Volume > 20-period average ===
    volume_ma = np.zeros_like(volume)
    for i in range(20, len(volume)):
        volume_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(weekly_pp_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(daily_high_aligned[i]) or 
            np.isnan(daily_low_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price relative to weekly pivot
        uptrend = close[i] > weekly_pp_aligned[i]
        downtrend = close[i] < weekly_pp_aligned[i]
        
        # Breakout conditions with volume confirmation
        breakout_long = (close[i] > daily_high_aligned[i]) and (volume[i] > volume_ma[i])
        breakdown_short = (close[i] < daily_low_aligned[i]) and (volume[i] > volume_ma[i])
        
        # Entry: breakout in direction of weekly trend
        long_signal = uptrend and breakout_long
        short_signal = downtrend and breakdown_short
        
        # Exit: trend reversal or opposite breakout
        exit_long = not uptrend or breakdown_short
        exit_short = not downtrend or breakout_long
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals