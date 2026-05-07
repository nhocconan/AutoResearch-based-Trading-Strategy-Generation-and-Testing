#!/usr/bin/env python3
name = "6h_WeeklyPivot_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    prev_weekly_high = df_weekly['high'].shift(1).values
    prev_weekly_low = df_weekly['low'].shift(1).values
    prev_weekly_close = df_weekly['close'].shift(1).values
    pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3
    r1 = 2 * pivot - prev_weekly_low
    s1 = 2 * pivot - prev_weekly_high
    
    # Align weekly pivot to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # Load daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Daily EMA(20) for trend filter
    ema_daily = pd.Series(df_daily['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Volume spike detection (24-period average for 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Wait for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_daily_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above weekly pivot with daily uptrend and volume spike
            if (close[i] > pivot_aligned[i] and 
                close[i] > ema_daily_aligned[i] and 
                volume[i] > vol_ma[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly pivot with daily downtrend and volume spike
            elif (close[i] < pivot_aligned[i] and 
                  close[i] < ema_daily_aligned[i] and 
                  volume[i] > vol_ma[i] * 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls below weekly pivot or trend changes
            if (close[i] < pivot_aligned[i] or 
                close[i] < ema_daily_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises above weekly pivot or trend changes
            if (close[i] > pivot_aligned[i] or 
                close[i] > ema_daily_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s weekly pivot + daily EMA(20) trend + volume confirmation.
# Weekly pivot provides institutional support/resistance from prior week's action.
# Trading in direction of daily EMA(20) ensures alignment with intermediate trend.
# Volume spike confirms institutional participation in the move.
# Works in bull markets (buying above pivot in uptrend) and bear markets (selling below pivot in downtrend).
# Position size 0.25 limits risk while allowing meaningful participation.