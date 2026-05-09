#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_Breakout_DailyTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    """
    6s Weekly Pivot Point breakout with daily trend filter.
    - Long: Close breaks above weekly R1 with price > daily EMA(34)
    - Short: Close breaks below weekly S1 with price < daily EMA(34)
    - Exit: Price crosses back through weekly pivot point
    - Uses actual weekly pivot levels from prior week
    - Target: 20-40 trades/year on 6h timeframe
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 40:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    close_daily = pd.Series(df_daily['close'].values)
    ema34_daily = close_daily.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Calculate weekly pivot point and levels
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Weekly pivot point
    pp_weekly = (high_weekly + low_weekly + close_weekly) / 3
    range_weekly = high_weekly - low_weekly
    r1_weekly = pp_weekly + (range_weekly * 1.1 / 12)
    s1_weekly = pp_weekly - (range_weekly * 1.1 / 12)
    
    # Align weekly levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp_weekly)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema34_daily_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above weekly R1 with daily uptrend
            if close[i] > r1_aligned[i] and close[i] > ema34_daily_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below weekly S1 with daily downtrend
            elif close[i] < s1_aligned[i] and close[i] < ema34_daily_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below weekly pivot
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above weekly pivot
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals