#!/usr/bin/env python3
"""
6h_WeeklyPivot_R3S3_Breakout_DailyTrend_VolumeSpike
Hypothesis: Weekly pivot R3/S3 levels act as strong support/resistance. Breakouts with daily trend (EMA34) and volume spike confirmation capture momentum in both bull and bear markets. Weekly timeframe reduces noise, daily trend filters direction, volume confirms conviction. Targets 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prrices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_daily = df_daily['close'].values
    ema_34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_34_daily)
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_34_daily_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Calculate weekly pivot levels for current week
        # Need previous week's OHLC (weekly data)
        week_idx = i // (7 * 24 * 60 // 6)  # 7 days * 24 hours * 60 minutes / 6 min per bar = 168 bars per week
        if week_idx < 1:
            signals[i] = 0.0
            continue
            
        prev_week_idx = week_idx - 1
        if prev_week_idx >= len(df_weekly):
            signals[i] = 0.0
            continue
            
        # Get previous week's OHLC from weekly data
        ph = df_weekly['high'].iloc[prev_week_idx]
        pl = df_weekly['low'].iloc[prev_week_idx]
        pc = df_weekly['close'].iloc[prev_week_idx]
        
        # Weekly pivot levels (standard calculation)
        pivot = (ph + pl + pc) / 3.0
        range_val = ph - pl
        r3 = pivot + range_val * 1.1
        s3 = pivot - range_val * 1.1
        
        # Trend direction from daily EMA34
        trend_up = close[i] > ema_34_daily_aligned[i]
        trend_down = close[i] < ema_34_daily_aligned[i]
        
        # Volume confirmation: >2.0x 20-period MA
        vol_confirm = volume[i] > (2.0 * vol_ma_20[i])
        
        # Breakout conditions
        long_breakout = close[i] > r3
        short_breakout = close[i] < s3
        
        # Entry logic
        long_entry = vol_confirm and trend_up and long_breakout
        short_entry = vol_confirm and trend_down and short_breakout
        
        # Exit logic: opposite breakout or trend reversal
        long_exit = (close[i] < s3) or (not trend_up)
        short_exit = (close[i] > r3) or (not trend_down)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_R3S3_Breakout_DailyTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0