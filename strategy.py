#!/usr/bin/env python3
"""
12h_WeeklyPivot_R1_S4_Breakout_1wTrend_Volume
Hypothesis: Use weekly trend filter (EMA34) with 1d Pivot R1/S4 breakout on 12h, requiring volume confirmation.
Pivot levels provide high-probability reversal points. Weekly EMA filter ensures trading with weekly trend.
Volume filter avoids false breakouts. Target: 15-25 trades/year, works in bull/bear via trend filter.
"""

name = "12h_WeeklyPivot_R1_S4_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily pivot points (using previous day)
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    pivot = (high_prev + low_prev + close_prev) / 3.0
    r1 = 2 * pivot - low_prev          # Resistance 1
    s4 = pivot - 3 * (high_prev - low_prev)  # Support 4 (strong support)
    
    # Align pivot levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Get 12h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 30-period EMA (moderate to balance trades)
    vol_ema30 = pd.Series(volume).ewm(span=30, adjust=False, min_periods=30).mean().values
    volume_filter = volume > vol_ema30 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA and previous day data
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend: price vs weekly EMA34
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: uptrend AND price breaks above daily R1 with volume
            if uptrend and high[i] > r1_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: downtrend AND price breaks below daily S4 with volume
            elif downtrend and low[i] < s4_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below daily S4 OR trend changes to downtrend
            if low[i] < s4_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above daily R1 OR trend changes to uptrend
            if high[i] > r1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals