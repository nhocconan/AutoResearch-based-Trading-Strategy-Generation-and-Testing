#!/usr/bin/env python3
# 6h_WeeklyPivot_Breakout_DailyTrend_Volume
# Hypothesis: Weekly pivot levels provide strong institutional support/resistance.
# Breakouts above weekly R1 in uptrend or below S1 in downtrend continue with momentum.
# Uses daily EMA34 for trend filter and volume confirmation to avoid false breakouts.
# Works in bull markets by following uptrends and bear markets by following downtrends.
# Target: 15-35 trades/year to minimize fee drag on 6h timeframe.

name = "6h_WeeklyPivot_Breakout_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 35:
        return np.zeros(n)
    
    # Calculate weekly pivot levels (using previous week's OHLC)
    weekly_high = df_weekly['high']
    weekly_low = df_weekly['low']
    weekly_close = df_weekly['close']
    pivot_point = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r1 = 2 * pivot_point - weekly_low
    weekly_s1 = 2 * pivot_point - weekly_high
    
    # Calculate daily EMA34 for trend filter
    ema_34_daily = pd.Series(df_daily['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly pivots and daily EMA to 6h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1.values)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1.values)
    ema_34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_34_daily)
    
    # Volume confirmation (24-period MA on 6h = 6 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly pivots (10), daily EMA34 (34), volume MA (24)
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(ema_34_daily_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_daily_aligned[i]
        downtrend = close[i] < ema_34_daily_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + price breaks above weekly R1 + volume
            if uptrend and close[i] > weekly_r1_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price breaks below weekly S1 + volume
            elif downtrend and close[i] < weekly_s1_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below weekly R1
            if not uptrend or close[i] < weekly_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters above weekly S1
            if not downtrend or close[i] > weekly_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals