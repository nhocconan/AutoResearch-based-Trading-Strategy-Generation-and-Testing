#!/usr/bin/env python3
"""
6h_WeeklyPivot_RangeBreakout_1dTrend_VolumeSpike
Hypothesis: Weekly pivot range breakouts on 6h timeframe with 1d trend filter and volume spike capture institutional flow in both bull and bear markets. Weekly pivots provide major support/resistance, while 1d trend filter reduces whipsaw and volume confirms breakout strength. Target: 15-30 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot levels from previous week
    # Weekly high/low/close: use last 5 trading days
    high_5d = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values
    low_5d = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values
    close_5d = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(1).values
    
    # Weekly pivot point and range
    pivot_weekly = (high_5d + low_5d + close_5d) / 3.0
    range_weekly = high_5d - low_5d
    
    # Weekly pivot support/resistance levels (standard calculation)
    r1_weekly = 2 * pivot_weekly - low_5d
    s1_weekly = 2 * pivot_weekly - high_5d
    r2_weekly = pivot_weekly + range_weekly
    s2_weekly = pivot_weekly - range_weekly
    
    # Align weekly levels to 6h timeframe
    r1_weekly_aligned = align_htf_to_ltf(prices, df_1d, r1_weekly)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_1d, s1_weekly)
    r2_weekly_aligned = align_htf_to_ltf(prices, df_1d, r2_weekly)
    s2_weekly_aligned = align_htf_to_ltf(prices, df_1d, s2_weekly)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for all indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_weekly_aligned[i]) or np.isnan(s1_weekly_aligned[i]) or 
            np.isnan(r2_weekly_aligned[i]) or np.isnan(s2_weekly_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions at weekly pivot levels
        breakout_long = close[i] > r1_weekly_aligned[i-1]  # Break above R1
        breakout_short = close[i] < s1_weekly_aligned[i-1]  # Break below S1
        
        # Strong breakout confirmation (beyond R2/S2 for institutional interest)
        strong_breakout_long = close[i] > r2_weekly_aligned[i-1]
        strong_breakout_short = close[i] < s2_weekly_aligned[i-1]
        
        # Trend filter from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions with volume confirmation and trend alignment
        long_entry = breakout_long and volume_spike[i] and uptrend
        short_entry = breakout_short and volume_spike[i] and downtrend
        
        # Strong breakout entries (higher conviction)
        strong_long_entry = strong_breakout_long and volume_spike[i] and uptrend
        strong_short_entry = strong_breakout_short and volume_spike[i] and downtrend
        
        # Exit on opposite breakout (reverse position)
        long_exit = breakout_short and volume_spike[i]
        short_exit = breakout_long and volume_spike[i]
        
        if (long_entry or strong_long_entry) and position <= 0:
            signals[i] = 0.25
            position = 1
        elif (short_entry or strong_short_entry) and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_RangeBreakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0