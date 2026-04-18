#!/usr/bin/env python3
"""
6h_Pivot_R1_S1_Breakout_With_Volume_and_WeeklyTrend
Hypothesis: Use weekly trend filter to avoid counter-trend trades in bear markets. Buy when price breaks above daily R1 with volume spike and weekly uptrend; short when breaks below daily S1 with volume spike and weekly downtrend. Weekly trend avoids whipsaws during 2022 crash and 2025 range market. Target: 50-150 trades over 4 years.
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
    volume = prices['volume'].values
    
    # Daily pivot levels for R1/S1
    df_1d = get_htf_data(prices, '1d')
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    rang = phigh - plow
    r1 = pclose + rang * 1.1 / 12
    s1 = pclose - rang * 1.1 / 12
    
    # Align daily levels to 6h (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Weekly trend: EMA50 on weekly close
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_trend_up = close_1w > ema_50_1w  # True if above weekly EMA50
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    
    # Volume spike: >1.8x 30-period average (more selective)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Need volume MA and weekly trend
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(weekly_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_spike = volume_spike[i]
        weekly_up = weekly_trend_aligned[i] > 0.5
        
        if position == 0:
            # Long: price > R1 with volume spike and weekly uptrend
            if price > r1_val and vol_spike and weekly_up:
                signals[i] = 0.25
                position = 1
            # Short: price < S1 with volume spike and weekly downtrend
            elif price < s1_val and vol_spike and not weekly_up:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < S1 or weekly trend turns down
            if price < s1_val or not weekly_up:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > R1 or weekly trend turns up
            if price > r1_val or weekly_up:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Pivot_R1_S1_Breakout_With_Volume_and_WeeklyTrend"
timeframe = "6h"
leverage = 1.0