#!/usr/bin/env python3
"""
1d_Weekly_Range_Breakout_Filter
Hypothesis: Breakouts from weekly range (Monday high/low) with volume confirmation and weekly trend filter.
Targets 15-30 trades/year on daily timeframe to minimize fee drag while capturing strong directional moves.
Uses weekly trend to filter direction, Monday range as breakout levels, and volume spike for confirmation.
Works in both bull and bear by following weekly trend direction.
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
    
    # Get weekly data for trend filter and Monday range
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Monday high/low from weekly data (weekly candles start at Monday 00:00 UTC)
    # For weekly data, the open is Monday 00:00, high/low are for the week
    # We'll use the weekly high/low as breakout levels (simplified approach)
    week_high = df_1w['high'].values
    week_low = df_1w['low'].values
    week_high_aligned = align_htf_to_ltf(prices, df_1w, week_high)
    week_low_aligned = align_htf_to_ltf(prices, df_1w, week_low)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for EMA20 and volume MA to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(week_high_aligned[i]) or
            np.isnan(week_low_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA20
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Volume confirmation (>1.5x average)
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Breakout conditions at weekly high/low
        long_breakout = close[i] > week_high_aligned[i] and vol_confirm and uptrend
        short_breakout = close[i] < week_low_aligned[i] and vol_confirm and downtrend
        
        # Exit conditions: return to weekly midpoint
        week_mid = (week_high_aligned[i] + week_low_aligned[i]) / 2
        long_exit = close[i] < week_mid
        short_exit = close[i] > week_mid
        
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

name = "1d_Weekly_Range_Breakout_Filter"
timeframe = "1d"
leverage = 1.0