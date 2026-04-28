#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Breakout_DailyTrend_Volume
Hypothesis: Uses weekly pivot points (from Monday's weekly open/close) with daily trend filter (EMA20) and volume spike (>1.5x average) to capture breakouts in line with weekly momentum. Weekly pivots provide stronger support/resistance than daily, reducing false breakouts. Works in bull/bear by following daily trend direction. Targets 20-30 trades/year via strict weekly pivot breakout conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    weekly_open = df_weekly['open'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    # Resistance 1 = (2 * P) - L
    r1 = (2 * pivot) - weekly_low
    # Support 1 = (2 * P) - H
    s1 = (2 * pivot) - weekly_high
    
    # Align weekly pivots to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # Daily trend filter: EMA20
    close_daily = df_daily['close'].values
    ema_20_daily = pd.Series(close_daily).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_20_daily)
    
    # Volume confirmation: >1.5x 48-period MA (2 days of 6h bars)
    vol_ma_48 = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 48  # Wait for volume MA to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_20_daily_aligned[i]) or
            np.isnan(vol_ma_48[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA20
        uptrend = close[i] > ema_20_daily_aligned[i]
        downtrend = close[i] < ema_20_daily_aligned[i]
        
        # Volume confirmation (>1.5x average)
        vol_confirm = volume[i] > (1.5 * vol_ma_48[i])
        
        # Breakout conditions at weekly R1/S1
        long_breakout = close[i] > r1_aligned[i] and vol_confirm and uptrend
        short_breakout = close[i] < s1_aligned[i] and vol_confirm and downtrend
        
        # Exit conditions: return to weekly pivot
        long_exit = close[i] < pivot_aligned[i]
        short_exit = close[i] > pivot_aligned[i]
        
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

name = "6h_Weekly_Pivot_Breakout_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0