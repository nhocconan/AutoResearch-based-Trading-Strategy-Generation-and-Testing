#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_WeeklyTrend
Hypothesis: Daily KAMA trend filter with weekly trend alignment for multi-timeframe confirmation.
Goes long when KAMA turns upward and weekly trend is bullish, short when KAMA turns downward and weekly trend is bearish.
Uses volume confirmation (>1.5x 20-day average) to filter false signals. Designed for very low trade frequency
(5-15 trades/year) to minimize fee decay while capturing major trend changes. Works in both bull and bear markets
by following the higher timeframe trend direction.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily KAMA ( Kaufman Adaptive Moving Average )
    # ER = Efficiency Ratio = abs(close - close[10]) / sum(abs(close - close[1])) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, axis=0)), axis=0)  # temporary fix, will replace below
    
    # Proper ER calculation
    change_over_10 = np.abs(close - np.roll(close, 10))
    sum_abs_change = np.zeros_like(close)
    for i in range(1, len(close)):
        sum_abs_change[i] = sum_abs_change[i-1] + np.abs(close[i] - close[i-1])
        if i >= 10:
            sum_abs_change[i] -= np.abs(close[i-10] - close[i-11]) if i >= 11 else 0
    
    # Handle first 10 values
    for i in range(10):
        sum_abs_change[i] = np.sum(np.abs(np.diff(close[:i+1])))
    
    # Avoid division by zero
    er = np.where(sum_abs_change > 0, change_over_10 / sum_abs_change, 0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for EMA 2
    slow_sc = 2 / (30 + 1)  # for EMA 30
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to daily timeframe (already daily, but ensure alignment)
    kama_aligned = kama  # already on daily timeframe
    
    # Volume confirmation: >1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(kama_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: slope of KAMA
        kama_rising = kama_aligned[i] > kama_aligned[i-1]
        kama_falling = kama_aligned[i] < kama_aligned[i-1]
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry logic: KAMA direction aligned with weekly trend and volume
        long_entry = vol_confirm and kama_rising and weekly_uptrend
        short_entry = vol_confirm and kama_falling and weekly_downtrend
        
        # Exit logic: opposite KAMA direction or weekly trend change
        long_exit = kama_falling or (not weekly_uptrend)
        short_exit = kama_rising or (not weekly_downtrend)
        
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

name = "1d_KAMA_Trend_Filter_WeeklyTrend"
timeframe = "1d"
leverage = 1.0