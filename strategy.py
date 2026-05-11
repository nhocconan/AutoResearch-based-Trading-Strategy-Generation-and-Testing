#!/usr/bin/env python3
"""
6h_Liquidity_Sweep_Reversal_WeeklyTrend
Hypothesis: Price sweeps above/below weekly high/low then reverses with volume confirmation. Works in both bull/bear as liquidity hunts precede reversals. Weekly trend filter ensures trades align with higher timeframe momentum. Target: 15-30 trades per year on 6h timeframe.
"""

name = "6h_Liquidity_Sweep_Reversal_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1W Data for Weekly High/Low and Trend ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly high/low (previous week's)
    prev_weekly_high = np.roll(high_1w, 1)
    prev_weekly_low = np.roll(low_1w, 1)
    prev_weekly_high[0] = high_1w[0]
    prev_weekly_low[0] = low_1w[0]
    
    # Weekly trend: EMA50 on weekly close
    weekly_ema50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly data to 6h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, prev_weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, prev_weekly_low)
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Volume confirmation: current volume > 1.5x 24-period average (4 days on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_surge = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or 
            np.isnan(weekly_ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: sweep below weekly low then reverse up with volume surge
            if low[i] < weekly_low_aligned[i] and close[i] > weekly_low_aligned[i] and volume_surge[i]:
                # Additional filter: weekly uptrend (price above weekly EMA50)
                if close[i] > weekly_ema50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Short: sweep above weekly high then reverse down with volume surge
            elif high[i] > weekly_high_aligned[i] and close[i] < weekly_high_aligned[i] and volume_surge[i]:
                # Additional filter: weekly downtrend (price below weekly EMA50)
                if close[i] < weekly_ema50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below weekly low OR weekly trend turns down
            if low[i] < weekly_low_aligned[i] or close[i] < weekly_ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks above weekly high OR weekly trend turns up
            if high[i] > weekly_high_aligned[i] or close[i] > weekly_ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals