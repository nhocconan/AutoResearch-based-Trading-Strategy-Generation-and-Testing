#!/usr/bin/env python3
"""
1d_WeeklyPivot_R3_S3_Breakout_WeeklyTrend
Hypothesis: Weekly R3/S1 breakout with weekly trend filter (EMA50) and volume confirmation.
Targets 10-20 trades/year on daily timeframe to minimize fee drift and capture major trends.
Works in bull/bear via trend filter and volatility-based breakout conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Calculate weekly pivot levels for current week
        # Need previous week's OHLC (weekly data)
        week_idx = i // (7 * 24 * 4)  # Approximate: 7 days * 24h * 4 (4h bars per day)
        if week_idx < 1:
            signals[i] = 0.0
            continue
            
        prev_week_idx = week_idx - 1
        if prev_week_idx >= len(df_1w):
            signals[i] = 0.0
            continue
            
        # Get previous week's OHLC from weekly data
        ph = df_1w['high'].iloc[prev_week_idx]
        pl = df_1w['low'].iloc[prev_week_idx]
        pc = df_1w['close'].iloc[prev_week_idx]
        
        # Weekly pivot levels (R3 and S3)
        range_val = ph - pl
        r3 = pc + (range_val * 1.1 * 2)  # R3 = C + 2*(H-L)*1.1
        s3 = pc - (range_val * 1.1 * 2)  # S3 = C - 2*(H-L)*1.1
        
        # Trend direction from weekly EMA50
        trend_up = close[i] > ema_50_1w_aligned[i]
        trend_down = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation: >2.0x 20-period MA
        vol_confirm = volume[i] > (2.0 * vol_ma_20[i])
        
        # Breakout conditions (R3 for long, S3 for short)
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

name = "1d_WeeklyPivot_R3_S3_Breakout_WeeklyTrend"
timeframe = "1d"
leverage = 1.0