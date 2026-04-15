#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot + Volume Spike + 1d EMA Trend Filter
# Uses weekly pivot points (R1/S1) for mean reversion in ranging markets.
# Long when price touches S1 with volume spike and 1d EMA > 1d EMA(50).
# Short when price touches R1 with volume spike and 1d EMA < 1d EMA(50).
# Weekly pivot provides strong support/resistance that works in both bull and bear markets.
# Volume spike confirms genuine interest at pivot levels.
# 1d EMA filter ensures we trade with higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot points
    df_week = get_htf_data(prices, '1w')
    if len(df_week) < 1:
        return np.zeros(n)
    week_high = df_week['high'].values
    week_low = df_week['low'].values
    week_close = df_week['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    week_pivot = (week_high + week_low + week_close) / 3.0
    week_r1 = 2 * week_pivot - week_low
    week_s1 = 2 * week_pivot - week_high
    
    # Align weekly pivot to 6h timeframe (weekly pivot is fixed for the week)
    week_pivot_aligned = align_htf_to_ltf(prices, df_week, week_pivot)
    week_r1_aligned = align_htf_to_ltf(prices, df_week, week_r1)
    week_s1_aligned = align_htf_to_ltf(prices, df_week, week_s1)
    
    # Load daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(20) and EMA(50) for trend filter
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 6h timeframe
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection: volume > 2x 20-period median
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median().values
    volume_spike = volume > (2.0 * vol_median)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(week_r1_aligned[i]) or np.isnan(week_s1_aligned[i]) or
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            continue
        
        # Long entry: price touches S1 with volume spike and bullish trend (EMA20 > EMA50)
        if (low[i] <= week_s1_aligned[i] and
            volume_spike[i] and
            ema_20_1d_aligned[i] > ema_50_1d_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price touches R1 with volume spike and bearish trend (EMA20 < EMA50)
        elif (high[i] >= week_r1_aligned[i] and
              volume_spike[i] and
              ema_20_1d_aligned[i] < ema_50_1d_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite pivot touch or trend reversal
        elif position == 1 and (high[i] >= week_r1_aligned[i] or ema_20_1d_aligned[i] < ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (low[i] <= week_s1_aligned[i] or ema_20_1d_aligned[i] > ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_Volume_EMATrend"
timeframe = "6h"
leverage = 1.0