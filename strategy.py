#!/usr/bin/env python3
"""
1d_WeeklyPivot_Breakout_RangeFilter
Hypothesis: Weekly pivot levels (mean of weekly high/low/close) act as strong support/resistance. Breakouts above/below pivot with weekly trend alignment (price > weekly EMA20) and range filter (weekly ATR < 50-day ATR median) capture trending moves while avoiding chop. Designed for low frequency (<20 trades/year) to minimize fee drag. Works in bull/bear by trading breakouts in direction of weekly trend.
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
    
    # Get weekly data for pivot and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot point: (H + L + C) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Weekly EMA20 for trend filter
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Weekly ATR(14) for volatility filter
    tr1 = weekly_high - weekly_low
    tr2 = np.abs(weekly_high - np.roll(weekly_close, 1))
    tr3 = np.abs(weekly_low - np.roll(weekly_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    weekly_atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 50-day ATR median for regime filter (using daily data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    tr1_d = daily_high - daily_low
    tr2_d = np.abs(daily_high - np.roll(daily_close, 1))
    tr3_d = np.abs(daily_low - np.roll(daily_close, 1))
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    tr_d[0] = tr1_d[0]
    daily_atr = pd.Series(tr_d).rolling(window=50, min_periods=50).mean().values
    # Calculate median of last 50 daily ATR values
    atr_median = np.full_like(daily_atr, np.nan)
    for i in range(49, len(daily_atr)):
        atr_median[i] = np.nanmedian(daily_atr[i-49:i+1])
    
    # Align weekly data to daily timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    weekly_atr_aligned = align_htf_to_ltf(prices, df_1w, weekly_atr)
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    atr_median_aligned = align_htf_to_ltf(prices, df_1d, atr_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(weekly_ema20_aligned[i]) or
            np.isnan(weekly_atr_aligned[i]) or
            np.isnan(daily_high_aligned[i]) or
            np.isnan(daily_low_aligned[i]) or
            np.isnan(atr_median_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Range filter: weekly ATR < 50-day ATR median (low volatility = avoid chop)
        low_volatility = weekly_atr_aligned[i] < atr_median_aligned[i]
        
        # Trend filter: price relative to weekly EMA20
        uptrend = close[i] > weekly_ema20_aligned[i]
        downtrend = close[i] < weekly_ema20_aligned[i]
        
        # Breakout conditions
        long_breakout = (close[i] > weekly_pivot_aligned[i]) and low_volatility and uptrend
        short_breakout = (close[i] < weekly_pivot_aligned[i]) and low_volatility and downtrend
        
        # Exit conditions: reverse when price crosses pivot in opposite direction
        long_exit = (position == 1) and (close[i] < weekly_pivot_aligned[i])
        short_exit = (position == -1) and (close[i] > weekly_pivot_aligned[i])
        
        if long_exit:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit:
            signals[i] = 0.25   # Reverse to long
            position = 1
        elif long_breakout and (position <= 0):
            signals[i] = 0.25
            position = 1
        elif short_breakout and (position >= 0):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyPivot_Breakout_RangeFilter"
timeframe = "1d"
leverage = 1.0