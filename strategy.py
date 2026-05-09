#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_DailyBreakout_TrendFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Previous weekly bar's OHLC (for pivot calculation)
    prev_close_weekly = df_weekly['close'].shift(1).values
    prev_high_weekly = df_weekly['high'].shift(1).values
    prev_low_weekly = df_weekly['low'].shift(1).values
    
    # Calculate weekly pivot points
    weekly_pivot = (prev_high_weekly + prev_low_weekly + prev_close_weekly) / 3
    weekly_range = prev_high_weekly - prev_low_weekly
    weekly_r1 = weekly_pivot + weekly_range * 1.0
    weekly_s1 = weekly_pivot - weekly_range * 1.0
    
    # Align weekly pivot points to daily
    weekly_pivot_daily = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    weekly_r1_daily = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_daily = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    
    # Get daily EMA34 for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    ema_34_daily = pd.Series(df_daily['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_daily, ema_34_daily)
    
    # Volume filter: above 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_r1_daily[i]) or np.isnan(weekly_s1_daily[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation
        
        if position == 0:
            # Long breakout: price breaks above weekly R1 with daily uptrend
            if (close[i] > weekly_r1_daily[i] and 
                close[i] > ema_34_aligned[i] and  # daily uptrend
                vol_ok):
                signals[i] = 0.30
                position = 1
            # Short breakdown: price breaks below weekly S1 with daily downtrend
            elif (close[i] < weekly_s1_daily[i] and 
                  close[i] < ema_34_aligned[i] and  # daily downtrend
                  vol_ok):
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly pivot (mean reversion)
            if close[i] < weekly_pivot_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: price rises back above weekly pivot (mean reversion)
            if close[i] > weekly_pivot_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals