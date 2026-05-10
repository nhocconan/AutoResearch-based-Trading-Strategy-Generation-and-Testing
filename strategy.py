#!/usr/bin/env python3
"""
6h_WeeklyPivot_Confluence_1dTrend_Volume
Hypothesis: Use weekly pivot points as key support/resistance levels on 6h timeframe, with daily trend filter and volume confirmation.
Weekly pivots provide institutional reference points that work across market regimes. 
Long when price breaks above weekly R1 with daily uptrend and volume spike.
Short when price breaks below weekly S1 with daily downtrend and volume spike.
Exit when price returns to weekly pivot point or trend reverses.
Designed for 50-150 total trades over 4 years (12-37/year) with controlled risk.
"""

name = "6h_WeeklyPivot_Confluence_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot points and daily data for trend filter
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 5 or len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    high_prev_week = df_1w['high'].shift(1).values
    low_prev_week = df_1w['low'].shift(1).values
    close_prev_week = df_1w['close'].shift(1).values
    
    # Weekly pivot calculation
    weekly_pivot = (high_prev_week + low_prev_week + close_prev_week) / 3.0
    weekly_r1 = 2 * weekly_pivot - low_prev_week
    weekly_s1 = 2 * weekly_pivot - high_prev_week
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 6h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2.0x 20-period EMA (to ensure significance)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly pivot data and EMA34
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend: price vs EMA34
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Long: uptrend AND price breaks above weekly R1 with volume spike
            if uptrend and high[i] > weekly_r1_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: downtrend AND price breaks below weekly S1 with volume spike
            elif downtrend and low[i] < weekly_s1_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to weekly pivot OR trend changes to downtrend
            if low[i] <= weekly_pivot_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to weekly pivot OR trend changes to uptrend
            if high[i] >= weekly_pivot_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals