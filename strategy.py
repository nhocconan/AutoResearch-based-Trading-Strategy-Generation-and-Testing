#!/usr/bin/env python3
"""
1d_Weekly_Pivot_Breakout_WeeklyTrend_VolumeFilter
Hypothesis: Uses weekly pivot points (R4/S4) with weekly trend filter (EMA34) and volume spike (>2x average) to capture strong breakouts. Works in bull/bear by following weekly trend direction. Targets 10-20 trades/year via strict weekly R4/S4 breakout conditions.
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
    
    # Get weekly data for pivot points and EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly pivot points (using previous week's data)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot point calculation
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    range_ = weekly_high - weekly_low
    R4 = pivot + (range_ * 1.1)  # R4 is at pivot + 1.1 * range
    S4 = pivot - (range_ * 1.1)  # S4 is at pivot - 1.1 * range
    
    # Align weekly pivot levels to daily timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1w, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1w, S4)
    
    # Volume confirmation: >2x 20-period MA (approx 4 weeks of daily bars)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for weekly EMA34 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(R4_aligned[i]) or
            np.isnan(S4_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA34
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Volume confirmation (>2x average)
        vol_confirm = volume[i] > (2.0 * vol_ma_20[i])
        
        # Breakout conditions at R4/S4
        long_breakout = close[i] > R4_aligned[i] and vol_confirm and uptrend
        short_breakout = close[i] < S4_aligned[i] and vol_confirm and downtrend
        
        # Exit conditions: return to weekly pivot point
        pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
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

name = "1d_Weekly_Pivot_Breakout_WeeklyTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0