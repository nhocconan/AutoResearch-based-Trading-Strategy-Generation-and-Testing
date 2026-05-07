#!/usr/bin/env python3
name = "1d_WeeklyPivot_R1S1_Breakout_Trend_Filter_v5"
timeframe = "1d"
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
    
    # Load weekly data ONCE before loop for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points using Monday's OHLC (start of week)
    # For each week, use the Monday of that week's OHLC
    days = pd.to_datetime(df_1w.index)
    week_start = days - pd.to_timedelta(days.weekday, unit='D')
    unique_weeks = week_start.unique()
    
    # Arrays to store weekly pivot levels for each week
    weekly_pivot = np.full(len(df_1w), np.nan)
    weekly_r1 = np.full(len(df_1w), np.nan)
    weekly_s1 = np.full(len(df_1w), np.nan)
    
    for week in unique_weeks:
        week_mask = week_start == week
        if np.sum(week_mask) == 0:
            continue
        # Get Monday (first day of week)
        monday_idx = np.where(week_mask)[0][0]
        if monday_idx >= len(df_1w):
            continue
        # Use Monday's OHLC for weekly pivot
        monday_high = df_1w['high'].iloc[monday_idx]
        monday_low = df_1w['low'].iloc[monday_idx]
        monday_close = df_1w['close'].iloc[monday_idx]
        
        pivot = (monday_high + monday_low + monday_close) / 3
        range_val = monday_high - monday_low
        
        r1 = pivot + (range_val * 1.1 / 12)
        s1 = pivot - (range_val * 1.1 / 12)
        
        # Assign to all days in this week
        weekly_pivot[week_mask] = pivot
        weekly_r1[week_mask] = r1
        weekly_s1[week_mask] = s1
    
    # Calculate daily EMA(34) for trend filter
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly pivot levels to daily timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Volume spike detection (20-period average on 1d)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA and EMA
    
    for i in range(start_idx, n):
        if np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or np.isnan(ema_34[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above weekly R1 with volume and in uptrend
            vol_condition = volume[i] > vol_ma[i] * 2.0
            uptrend = close[i] > ema_34[i]
            
            if close[i] > weekly_r1_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with volume and in downtrend
            elif close[i] < weekly_s1_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below weekly pivot or volume drops
            if close[i] < weekly_pivot_aligned[i] or volume[i] < vol_ma[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above weekly pivot or volume drops
            if close[i] > weekly_pivot_aligned[i] or volume[i] < vol_ma[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 1d Weekly Pivot (using Monday OHLC) + Daily EMA(34) trend + Volume Spike (2x)
# Weekly pivot from Monday's OHLC provides significant weekly support/resistance.
# Breaks above R1 or below S1 with 2x volume indicate institutional interest.
# Daily EMA(34) ensures trades align with daily trend direction.
# Works in bull (buy R1 breaks in uptrend) and bear (sell S1 breaks in downtrend).
# Position size 0.25 balances risk and keeps trade frequency ~10-25/year.