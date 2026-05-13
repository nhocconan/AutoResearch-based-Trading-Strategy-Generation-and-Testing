#!/usr/bin/env python3
"""
1d_WeeklyPivot_HighLow_Breakout_TrendFilter
Hypothesis: Breakouts above weekly pivot resistance (R1) in uptrend (price > weekly EMA20) and breakdowns below weekly pivot support (S1) in downtrend (price < weekly EMA20) with volume confirmation (volume > 1.5x 20-day average). Designed for 1d timeframe to capture multi-day trends in both bull and bear markets with low trade frequency.
"""

name = "1d_WeeklyPivot_HighLow_Breakout_TrendFilter"
timeframe = "1d"
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
    
    # Get weekly data for pivot levels and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (based on previous week's OHLC)
    prev_high = df_weekly['high'].shift(1).values
    prev_low = df_weekly['low'].shift(1).values
    prev_close = df_weekly['close'].shift(1).values
    
    valid_idx = ~np.isnan(prev_high) & ~np.isnan(prev_low) & ~np.isnan(prev_close)
    pivot_point = np.full_like(prev_close, np.nan)
    resistance_r1 = np.full_like(prev_close, np.nan)
    support_s1 = np.full_like(prev_close, np.nan)
    
    pivot_point[valid_idx] = (prev_high[valid_idx] + prev_low[valid_idx] + prev_close[valid_idx]) / 3.0
    resistance_r1[valid_idx] = 2.0 * pivot_point[valid_idx] - prev_low[valid_idx]
    support_s1[valid_idx] = 2.0 * pivot_point[valid_idx] - prev_high[valid_idx]
    
    # Align weekly pivot levels to daily timeframe
    resistance_r1_aligned = align_htf_to_ltf(prices, df_weekly, resistance_r1)
    support_s1_aligned = align_htf_to_ltf(prices, df_weekly, support_s1)
    pivot_point_aligned = align_htf_to_ltf(prices, df_weekly, pivot_point)
    
    # Weekly EMA20 for trend filter
    ema_20_weekly = pd.Series(df_weekly['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Price breaks above R1 with volume confirmation in uptrend (price > weekly EMA20)
            if (not np.isnan(resistance_r1_aligned[i]) and 
                high[i] > resistance_r1_aligned[i] and 
                volume_confirmed[i] and 
                close[i] > ema_20_weekly_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume confirmation in downtrend (price < weekly EMA20)
            elif (not np.isnan(support_s1_aligned[i]) and 
                  low[i] < support_s1_aligned[i] and 
                  volume_confirmed[i] and 
                  close[i] < ema_20_weekly_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below pivot point or trend weakens
            if (not np.isnan(pivot_point_aligned[i]) and 
                low[i] < pivot_point_aligned[i]) or \
               (not np.isnan(ema_20_weekly_aligned[i]) and 
                close[i] < ema_20_weekly_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above pivot point or trend weakens
            if (not np.isnan(pivot_point_aligned[i]) and 
                high[i] > pivot_point_aligned[i]) or \
               (not np.isnan(ema_20_weekly_aligned[i]) and 
                close[i] > ema_20_weekly_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals