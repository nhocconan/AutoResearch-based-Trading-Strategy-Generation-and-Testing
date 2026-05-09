#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_DailyBreakout_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    prev_weekly_close = df_1w['close'].shift(1).values
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    
    # Weekly pivot: (H + L + C) / 3
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3
    
    # Weekly support/resistance levels
    R1 = 2 * weekly_pivot - prev_weekly_low
    S1 = 2 * weekly_pivot - prev_weekly_high
    R2 = weekly_pivot + (prev_weekly_high - prev_weekly_low)
    S2 = weekly_pivot - (prev_weekly_high - prev_weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    R1_6h = align_htf_to_ltf(prices, df_1w, R1)
    S1_6h = align_htf_to_ltf(prices, df_1w, S1)
    R2_6h = align_htf_to_ltf(prices, df_1w, R2)
    S2_6h = align_htf_to_ltf(prices, df_1w, S2)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: above 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_6h[i]) or np.isnan(R1_6h[i]) or np.isnan(S1_6h[i]) or
            np.isnan(R2_6h[i]) or np.isnan(S2_6h[i]) or np.isnan(ema_50_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.3 * vol_ma[i]  # Volume confirmation
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long breakout: price breaks above weekly R1 with daily uptrend
            if (close[i] > R1_6h[i] and 
                close[i] > ema_50_6h[i] and  # Daily uptrend
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below weekly S1 with daily downtrend
            elif (close[i] < S1_6h[i] and 
                  close[i] < ema_50_6h[i] and  # Daily downtrend
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly pivot (mean reversion to pivot)
            if close[i] < weekly_pivot_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly pivot (mean reversion to pivot)
            if close[i] > weekly_pivot_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals