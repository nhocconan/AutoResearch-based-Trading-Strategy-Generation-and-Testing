#!/usr/bin/env python3
"""
6h_WeeklyPivot_DailyTrend_Breakout
Hypothesis: Price breaking above/below weekly pivot levels (R1/S1) with daily trend alignment and volume confirmation
captures institutional breakouts that work in both bull and bear markets. Uses weekly structure for direction,
daily EMA for trend filter, and volume to avoid false breakouts. Targets 15-35 trades/year.
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
    
    # Volume confirmation (20-period MA)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    pivot_weekly = (high_weekly + low_weekly + close_weekly) / 3.0
    r1_weekly = 2 * pivot_weekly - low_weekly
    s1_weekly = 2 * pivot_weekly - high_weekly
    
    # Align weekly pivot levels to 6h timeframe
    r1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s1_weekly)
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    close_daily = df_daily['close'].values
    
    # Calculate daily EMA50 for trend filter
    close_series_daily = pd.Series(close_daily)
    ema50_daily = close_series_daily.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA to 6h timeframe
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 50)  # volume MA20, EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(r1_weekly_aligned[i]) or 
            np.isnan(s1_weekly_aligned[i]) or 
            np.isnan(ema50_daily_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long: price > weekly R1 + volume filter + daily uptrend
            if close[i] > r1_weekly_aligned[i] and volume_filter and close[i] > ema50_daily_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < weekly S1 + volume filter + daily downtrend
            elif close[i] < s1_weekly_aligned[i] and volume_filter and close[i] < ema50_daily_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < weekly S1 or daily trend turns down
            if close[i] < s1_weekly_aligned[i] or close[i] < ema50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > weekly R1 or daily trend turns up
            if close[i] > r1_weekly_aligned[i] or close[i] > ema50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_DailyTrend_Breakout"
timeframe = "6h"
leverage = 1.0