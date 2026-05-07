#!/usr/bin/env python3
# 6H_WeeklyPivot_R1S1_Breakout_1DTrend_Volume
# Hypothesis: Combines weekly pivot R1/S1 breakout with 1-day EMA trend filter and volume confirmation.
# Uses weekly pivots for long-term structure and 1-day EMA for intermediate trend, reducing whipsaw.
# Designed for 6h timeframe with low trade frequency (<30/year) to avoid fee drag in bear markets.
# Target: 15-30 trades per year per symbol with clear entry/exit rules.

name = "6H_WeeklyPivot_R1S1_Breakout_1DTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly OHLC for pivots
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Weekly pivot points: R1 = 2*P - L, S1 = 2*P - H
    pivot = (high_weekly + low_weekly + close_weekly) / 3
    weekly_r1 = 2 * pivot - low_weekly
    weekly_s1 = 2 * pivot - high_weekly
    
    # Get daily data for EMA trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # 1-day EMA34 for trend filter
    close_daily = df_daily['close'].values
    ema34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly pivots and daily EMA to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    ema34_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Volume filter: current volume > 1.5x average volume (30-period)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 34)  # Ensure we have volume MA and EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above R1 + Uptrend (price > EMA34) + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema34_aligned[i] and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + Downtrend (price < EMA34) + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema34_aligned[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit conditions:
            # 1. Price returns to weekly pivot (mean reversion)
            # 2. Trend change: price crosses EMA34 against position
            pivot_reversion = (position == 1 and close[i] < pivot[i]) or (position == -1 and close[i] > pivot[i])
            trend_change = (position == 1 and close[i] < ema34_aligned[i]) or (position == -1 and close[i] > ema34_aligned[i])
            
            if pivot_reversion or trend_change:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals