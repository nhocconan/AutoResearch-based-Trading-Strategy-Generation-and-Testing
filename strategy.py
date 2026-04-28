#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Breakout_DailyTrend_VolumeFilter
Hypothesis: Combines weekly pivot point breakouts with daily trend filter (EMA50) and volume confirmation (>1.8x average) to capture strong directional moves. Works in bull/bear by following daily trend direction. Weekly pivots provide stronger support/resistance than daily, reducing false breakouts. Targets 10-20 trades/year via strict weekly pivot breakout conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate weekly pivot points (using prior week's data)
    # Group daily data into weeks: Monday=0, Sunday=6
    df_1d = df_1d.copy()
    df_1d['week_start'] = df_1d.index - pd.to_timedelta(df_1d.index.weekday, unit='D')
    weekly = df_1d.groupby('week_start').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(weekly) < 2:
        return np.zeros(n)
    
    # Calculate pivot points for each week (using prior week's data)
    weekly['pp'] = (weekly['high'].shift(1) + weekly['low'].shift(1) + weekly['close'].shift(1)) / 3
    weekly['r1'] = 2 * weekly['pp'] - weekly['low'].shift(1)
    weekly['s1'] = 2 * weekly['pp'] - weekly['high'].shift(1)
    weekly['r2'] = weekly['pp'] + (weekly['high'].shift(1) - weekly['low'].shift(1))
    weekly['s2'] = weekly['pp'] - (weekly['high'].shift(1) - weekly['low'].shift(1))
    
    # Forward fill weekly values to daily index
    weekly_indexed = weekly.set_index('week_start')
    # Reindex to daily frequency, forward filling weekly values
    daily_index = df_1d.index
    pp_daily = weekly_indexed['pp'].reindex(daily_index, method='ffill').values
    r1_daily = weekly_indexed['r1'].reindex(daily_index, method='ffill').values
    s1_daily = weekly_indexed['s1'].reindex(daily_index, method='ffill').values
    r2_daily = weekly_indexed['r2'].reindex(daily_index, method='ffill').values
    s2_daily = weekly_indexed['s2'].reindex(daily_index, method='ffill').values
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_daily)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_daily)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_daily)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_daily)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_daily)
    
    # Volume confirmation: >1.8x 30-period MA (5 days of 6h bars)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 and weekly data to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(pp_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation (>1.8x average)
        vol_confirm = volume[i] > (1.8 * vol_ma_30[i])
        
        # Breakout conditions at weekly R2/S2 (stronger breakout signals)
        long_breakout = close[i] > r2_aligned[i] and vol_confirm and uptrend
        short_breakout = close[i] < s2_aligned[i] and vol_confirm and downtrend
        
        # Exit conditions: return to weekly pivot point
        long_exit = close[i] < pp_aligned[i]
        short_exit = close[i] > pp_aligned[i]
        
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
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

name = "6h_Weekly_Pivot_Breakout_DailyTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0