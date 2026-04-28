#!/usr/bin/env python3
"""
1d_WeeklyPivot_R3_S3_Breakout_WeeklyTrend_Volume
Hypothesis: Weekly Camarilla pivot (R3/S3) breakout on daily timeframe with weekly EMA20 trend filter and volume spike confirmation.
Targets 10-25 trades/year to minimize fee drift while capturing strong trend moves in both bull and bear markets.
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
    
    # Get weekly data for Camarilla pivot calculation and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_weekly = df_weekly['close'].values
    ema_20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_20_weekly_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Calculate weekly Camarilla pivot levels
        # Need previous week's OHLC (weekly data)
        week_idx = i // 480  # 480 = 20*24 (20 days per month approximation for weekly)
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
        
        # Camarilla levels (R3 and S3)
        range_val = ph - pl
        r3 = pc + (range_val * 1.1 / 4)  # R3 level
        s3 = pc - (range_val * 1.1 / 4)  # S3 level
        
        # Trend direction from weekly EMA20
        trend_up = close[i] > ema_20_weekly_aligned[i]
        trend_down = close[i] < ema_20_weekly_aligned[i]
        
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

name = "1d_WeeklyPivot_R3_S3_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0