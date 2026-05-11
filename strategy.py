#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Daily_Trend_Filter_v1
Hypothesis: Uses weekly pivot points for directional bias and daily trend confirmation.
In bullish weekly bias (price above weekly pivot), we take long positions when daily EMA21 is above EMA50.
In bearish weekly bias (price below weekly pivot), we take short positions when daily EMA21 is below EMA50.
Designed for low trade frequency (~15-25 trades/year) by requiring alignment of weekly structure and daily trend.
Works in both bull and bear markets by following the weekly structure while using daily trend for entry timing.
"""

name = "6h_Weekly_Pivot_Daily_Trend_Filter_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Get daily data for trend
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- Weekly Pivot Points ---
    # Using previous week's OHLC to calculate current week's pivot
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate pivot point: (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    
    # --- Daily EMA Trend ---
    # Calculate EMA21 and EMA50 on daily close
    close_daily = df_daily['close'].values
    ema21_daily = pd.Series(close_daily).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMAs to 6h
    ema21_daily_aligned = align_htf_to_ltf(prices, df_daily, ema21_daily)
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(ema21_daily_aligned[i]) or 
            np.isnan(ema50_daily_aligned[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine weekly bias
        weekly_bullish = close[i] > weekly_pivot_aligned[i]
        weekly_bearish = close[i] < weekly_pivot_aligned[i]
        
        # Determine daily trend
        daily_uptrend = ema21_daily_aligned[i] > ema50_daily_aligned[i]
        daily_downtrend = ema21_daily_aligned[i] < ema50_daily_aligned[i]
        
        if position == 0:
            # Enter long: weekly bullish + daily uptrend
            if weekly_bullish and daily_uptrend:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly bearish + daily downtrend
            elif weekly_bearish and daily_downtrend:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: weekly bias changes or daily trend reverses
            if position == 1:
                # Exit long: weekly turns bearish OR daily turns downtrend
                exit_signal = (not weekly_bullish) or (not daily_uptrend)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: weekly turns bullish OR daily turns uptrend
                exit_signal = (not weekly_bearish) or (not daily_downtrend)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals