#!/usr/bin/env python3
"""
1d_1w_WeeklyPivot_DonchianBreakout_v1
Hypothesis: Breakout above weekly resistance (high of prior week) with price above daily pivot (bullish) or below weekly support (low of prior week) with price below daily pivot (bearish), filtered by weekly ADX > 25 for trending conditions. Weekly timeframe provides strong trend context, daily timeframe provides entry precision. Designed for low-frequency, high-conviction trades in both bull and bear markets.
Target: 20-60 total trades over 4 years (5-15/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_WeeklyPivot_DonchianBreakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === WEEKLY DATA ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point (standard formula)
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Weekly support/resistance (prior week high/low for breakout)
    weekly_high = high_1w  # current week high
    weekly_low = low_1w    # current week low
    
    # Align weekly data to daily
    pivot_daily = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_high_daily = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_daily = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # === WEEKLY ADX FOR TREND FILTER ===
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align indices
    
    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period]) if not np.all(np.isnan(data[1:period])) else np.nan
        # Subsequent values: smoothed = (prev_smoothed * (period-1) + current) / period
        for i in range(period, len(data)):
            if np.isnan(result[i-1]) or np.isnan(data[i]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    adx_daily = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(pivot_daily[i]) or np.isnan(weekly_high_daily[i]) or 
            np.isnan(weekly_low_daily[i]) or np.isnan(adx_daily[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: break above weekly high with price above weekly pivot and ADX > 25
        long_signal = (high[i] > weekly_high_daily[i] and 
                      close[i] > pivot_daily[i] and 
                      adx_daily[i] > 25)
        
        # Short: break below weekly low with price below weekly pivot and ADX > 25
        short_signal = (low[i] < weekly_low_daily[i] and 
                       close[i] < pivot_daily[i] and 
                       adx_daily[i] > 25)
        
        # Exit: price crosses back through weekly pivot
        exit_long = (position == 1 and close[i] < pivot_daily[i])
        exit_short = (position == -1 and close[i] > pivot_daily[i])
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals