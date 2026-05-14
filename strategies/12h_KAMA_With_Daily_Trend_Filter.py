#!/usr/bin/env python3
"""
12h_KAMA_With_Daily_Trend_Filter
Hypothesis: Trade 12h KAMA direction with daily trend filter. Long when KAMA rising + daily uptrend; short when KAMA falling + daily downtrend.
KAMA adapts to market noise, reducing whipsaw in ranging markets. Daily trend filter ensures alignment with higher timeframe momentum.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
Works in bull/bear: daily filter avoids counter-trend trades, KAMA reduces false signals in chop.
"""

name = "12h_KAMA_With_Daily_Trend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 10:
        return np.zeros(n)
    
    # Calculate daily EMA20 for trend filter
    close_daily = df_daily['close'].values
    ema20_daily = np.full_like(close_daily, np.nan)
    if len(close_daily) >= 20:
        multiplier = 2.0 / (20 + 1)
        ema20_daily[19] = np.mean(close_daily[:20])
        for i in range(20, len(close_daily)):
            ema20_daily[i] = multiplier * close_daily[i] + (1 - multiplier) * ema20_daily[i-1]
    ema20_daily_aligned = align_htf_to_ltf(prices, df_daily, ema20_daily)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = |net change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    direction = np.abs(np.subtract(close, np.roll(close, 1)))
    direction[0] = 0  # first element
    
    er = np.zeros(n)
    for i in range(2, n):  # need at least 2 periods
        if np.sum(change[i-9:i+1]) > 0:  # 10-period ER
            er[i] = direction[i] / np.sum(change[i-9:i+1])
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2.0/(2+1) - 2.0/(30+1)) + 2.0/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema20_daily_aligned[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA rising + daily uptrend
            if kama[i] > kama[i-1] and close[i] > ema20_daily_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling + daily downtrend
            elif kama[i] < kama[i-1] and close[i] < ema20_daily_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA falling OR daily trend turns down
            if kama[i] < kama[i-1] or close[i] < ema20_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA rising OR daily trend turns up
            if kama[i] > kama[i-1] or close[i] > ema20_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals