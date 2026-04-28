#!/usr/bin/env python3
"""
1d_1w_Pivot_Breakout_Trend_Filter
Hypothesis: Uses weekly pivot points (R4/S4) from weekly chart with daily trend filter (EMA50) and volume confirmation to capture strong breakouts. Weekly pivots provide stronger support/resistance than daily, reducing false signals. Trend filter ensures alignment with higher timeframe momentum. Volume confirmation adds conviction. Designed for low trade frequency (<25/year) to minimize fee drag, suitable for both bull and bear markets by following trend direction.
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
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    # Standard pivot: (H + L + C) / 3
    # R4 = C + 3*(H - L), S4 = C - 3*(H - L)
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    pivot_point = (prev_week_high + prev_week_low + prev_week_close) / 3
    weekly_range = prev_week_high - prev_week_low
    R4 = pivot_point + (3 * weekly_range)
    S4 = pivot_point - (3 * weekly_range)
    
    # Align weekly R4/S4 to daily timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1w, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1w, S4)
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: >1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(R4_aligned[i]) or
            np.isnan(S4_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation (>1.5x average)
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Breakout conditions at weekly R4/S4
        long_breakout = close[i] > R4_aligned[i] and vol_confirm and uptrend
        short_breakout = close[i] < S4_aligned[i] and vol_confirm and downtrend
        
        # Exit conditions: return to weekly pivot point
        long_exit = close[i] < pivot_point[i]  # Using previous week's pivot
        short_exit = close[i] > pivot_point[i]
        
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

name = "1d_1w_Pivot_Breakout_Trend_Filter"
timeframe = "1d"
leverage = 1.0