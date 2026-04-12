#!/usr/bin/env python3
"""
6h_1w_Pivot_Reversal_Confluence_v1
Hypothesis: Trade reversals at weekly pivot levels with 1d trend filter and volume confirmation. 
In bull markets: buy at weekly S1/S2 in uptrend. In bear markets: sell at weekly R1/R2 in downtrend.
Weekly pivots provide strong institutional levels; 1d EMA50 filters trend direction; volume spike confirms conviction.
Targets 15-25 trades/year by requiring confluence of weekly support/resistance, trend alignment, and volume.
Works in both bull (buying dips in uptrend) and bear (selling rallies in downtrend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_Pivot_Reversal_Confluence_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average (20 period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly pivot points (HIGH, LOW, CLOSE from previous week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_point = (high_1w + low_1w + close_1w) / 3
    pivot_range = high_1w - low_1w
    
    # Weekly support/resistance levels
    r1 = 2 * pivot_point - low_1w
    s1 = 2 * pivot_point - high_1w
    r2 = pivot_point + pivot_range
    s2 = pivot_point - pivot_range
    r3 = high_1w + 2 * (pivot_point - low_1w)
    s3 = low_1w - 2 * (high_1w - pivot_point)
    
    # Align weekly levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(vol_ma[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend direction from daily EMA50
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Volume spike confirmation
        volume_spike = volume[i] > vol_ma[i] * 1.8
        
        # Rejection at weekly resistance (sell signal)
        resistance_rejection = (
            (close[i] < r1_aligned[i] and low[i] > r1_aligned[i]) or  # Pin bar rejection
            (close[i] < r2_aligned[i] and low[i] > r2_aligned[i]) or
            (close[i] < r3_aligned[i] and low[i] > r3_aligned[i])
        )
        
        # Rejection at weekly support (buy signal)
        support_rejection = (
            (close[i] > s1_aligned[i] and high[i] < s1_aligned[i]) or  # Pin bar rejection
            (close[i] > s2_aligned[i] and high[i] < s2_aligned[i]) or
            (close[i] > s3_aligned[i] and high[i] < s3_aligned[i])
        )
        
        # Entry conditions
        long_entry = support_rejection and uptrend and volume_spike
        short_entry = resistance_rejection and downtrend and volume_spike
        
        # Exit conditions: opposite signal or middle of range
        long_exit = resistance_rejection or close[i] > pivot_point[i] if not np.isnan(pivot_point[i]) else False
        short_exit = support_rejection or close[i] < pivot_point[i] if not np.isnan(pivot_point[i]) else False
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals