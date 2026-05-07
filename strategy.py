#!/usr/bin/env python3
"""
12h_WeeklyPivot_DailyTrend_Volume
Hypothesis: Weekly pivot levels (resistance/support) act as strong institutional barriers in ranging and trending markets.
We combine weekly pivot points (from weekly OHLC) with daily trend (EMA50) and volume confirmation to capture
mean-reversion bounces off pivot levels while avoiding counter-trend traps. In bull markets, we buy near weekly S1/S2
with daily uptrend; in bear markets, we sell near weekly R1/R2 with daily downtrend. Weekly timeframe reduces noise,
daily trend filters direction, and volume confirms conviction. Target: 20-50 trades over 4 years.
"""
name = "12h_WeeklyPivot_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (using weekly OHLC)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    
    # Align weekly pivots to 12h timeframe (wait for weekly bar to close)
    pivot_aligned = align_ltf_to_htf(prices, df_weekly, pivot)
    r1_aligned = align_ltf_to_htf(prices, df_weekly, r1)
    s1_aligned = align_ltf_to_htf(prices, df_weekly, s1)
    r2_aligned = align_ltf_to_htf(prices, df_weekly, r2)
    s2_aligned = align_ltf_to_htf(prices, df_weekly, s2)
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema_50_daily = pd.Series(df_daily['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_daily_aligned = align_ltf_to_htf(prices, df_daily, ema_50_daily)
    
    # Volume filter: current volume > 1.3 * 24-period average (24*12h = 12 days)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for daily EMA
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema_50_daily_aligned[i]) or
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price near weekly support (S1 or S2) + daily uptrend + volume
            near_support = (close[i] <= s1_aligned[i] * 1.02) or (close[i] <= s2_aligned[i] * 1.02)
            if near_support and close[i] > ema_50_daily_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price near weekly resistance (R1 or R2) + daily downtrend + volume
            elif (close[i] >= r1_aligned[i] * 0.98) or (close[i] >= r2_aligned[i] * 0.98):
                if close[i] < ema_50_daily_aligned[i] and volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
        elif position != 0:
            # Exit: price returns to weekly pivot (mean reversion to pivot level)
            if position == 1:
                if close[i] >= pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] <= pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals