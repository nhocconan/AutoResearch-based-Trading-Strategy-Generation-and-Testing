#!/usr/bin/env python3
"""
6h_WeeklyPivot_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Weekly pivot points (R3/S3) breakout on 6h with 1d EMA34 trend filter and volume spike confirmation.
Uses weekly pivots calculated from weekly OHLC, with 1d EMA34 as trend filter and volume > 2x 20-period MA for confirmation.
Targets 15-30 trades/year to minimize fee drift while capturing major trend moves.
Works in both bull and bear markets by following 1d trend direction.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Calculate weekly pivot levels for current week
        # Need previous week's OHLC (1w data)
        week_idx = i // 28  # 28 = 7*4 (6h bars per week)
        if week_idx < 1:
            signals[i] = 0.0
            continue
            
        prev_week_idx = week_idx - 1
        if prev_week_idx >= len(df_1w):
            signals[i] = 0.0
            continue
            
        # Get previous week's OHLC from 1w data
        ph = df_1w['high'].iloc[prev_week_idx]
        pl = df_1w['low'].iloc[prev_week_idx]
        pc = df_1w['close'].iloc[prev_week_idx]
        
        # Weekly pivot points calculation
        pivot = (ph + pl + pc) / 3.0
        r3 = pc + 1.1 * (ph - pl)
        s3 = pc - 1.1 * (ph - pl)
        
        # Trend direction from 1d EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation: >2.0x 20-period MA
        vol_confirm = volume[i] > (2.0 * vol_ma_20[i])
        
        # Breakout conditions at R3/S3
        long_breakout = close[i] > r3
        short_breakout = close[i] < s3
        
        # Entry logic
        long_entry = vol_confirm and trend_up and long_breakout
        short_entry = vol_confirm and trend_down and short_breakout
        
        # Exit logic: opposite breakout or trend reversal
        long_exit = (close[i] < pivot) or (not trend_up)
        short_exit = (close[i] > pivot) or (not trend_down)
        
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

name = "6h_WeeklyPivot_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0