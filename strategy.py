#!/usr/bin/env python3
# 1d_PivotPoints_WeeklyTrend_Volume
# Hypothesis: Use weekly pivot points as key support/resistance levels with daily price action.
# Long when price crosses above weekly pivot level with daily close > weekly EMA34 and volume > 1.5x average.
# Short when price crosses below weekly pivot level with daily close < weekly EMA34 and volume > 1.5x average.
# Exit when price crosses back through the weekly pivot level.
# Designed to capture institutional level reactions with trend and volume filters, effective in both trending and ranging markets.
# Targets 15-25 trades/year to minimize fee drag on daily timeframe.

name = "1d_PivotPoints_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly high, low, close for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly pivot point: (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to daily timeframe (available after weekly close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Weekly EMA34 for trend filter
    weekly_ema = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Volume confirmation: 20-day moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_ema_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price crosses above weekly pivot with daily close > weekly EMA and volume > 1.5x MA
            if close[i] > weekly_pivot_aligned[i] and close[i-1] <= weekly_pivot_aligned[i-1] and \
               close[i] > weekly_ema_aligned[i] and volume[i] > vol_ma[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below weekly pivot with daily close < weekly EMA and volume > 1.5x MA
            elif close[i] < weekly_pivot_aligned[i] and close[i-1] >= weekly_pivot_aligned[i-1] and \
                 close[i] < weekly_ema_aligned[i] and volume[i] > vol_ma[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below weekly pivot
            if close[i] < weekly_pivot_aligned[i] and close[i-1] >= weekly_pivot_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above weekly pivot
            if close[i] > weekly_pivot_aligned[i] and close[i-1] <= weekly_pivot_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals