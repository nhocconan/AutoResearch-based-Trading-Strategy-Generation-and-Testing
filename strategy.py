#!/usr/bin/env python3
"""
6h_Weekly_Pivot_1dTrend_Volume_Confirmation
Hypothesis: Weekly pivot levels provide institutional support/resistance. 
Breakouts above weekly R1/S1 with daily trend and volume confirmation capture 
multi-day momentum while avoiding false signals. Works in bull/bear markets 
by trading breakouts in direction of higher timeframe trend.
Target: 20-35 trades/year per symbol to avoid fee drag.
"""

name = "6h_Weekly_Pivot_1dTrend_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0

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
    
    # Weekly pivot points (R1, S1, R2, S2) - calculated from prior week
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Pivot point calculation
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = (2 * pivot) - weekly_low
    s1 = (2 * pivot) - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    
    # Align weekly pivots to 6h timeframe (only use after weekly bar closes)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # Daily trend filter - 1d EMA50
    df_daily = get_htf_data(prices, '1d')
    ema_50_daily = pd.Series(df_daily['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_daily, ema_50_daily)
    daily_uptrend = close > ema_50_aligned
    daily_downtrend = close < ema_50_aligned
    
    # Volume confirmation: > 1.3x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if position == 0:
            # LONG: break above weekly R1 with daily uptrend and volume confirmation
            if close[i] > r1_aligned[i] and daily_uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below weekly S1 with daily downtrend and volume confirmation
            elif close[i] < s1_aligned[i] and daily_downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls back to weekly pivot or trend reverses
            if close[i] < pivot_aligned[i] or not daily_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises back to weekly pivot or trend reverses
            if close[i] > pivot_aligned[i] or not daily_downtrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals