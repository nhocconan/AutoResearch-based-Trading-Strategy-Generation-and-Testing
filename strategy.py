#!/usr/bin/env python3
name = "12h_WeeklyPivot_Breakout_Trend_Volume"
timeframe = "12h"
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
    
    # Load daily data ONCE before loop for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points using Monday's OHLC (start of week)
    days = pd.to_datetime(df_1d.index)
    week_start = days - pd.to_timedelta(days.weekday, unit='D')
    unique_weeks = week_start.unique()
    
    # Arrays to store weekly pivot levels for each day
    weekly_pivot = np.full(len(df_1d), np.nan)
    weekly_r1 = np.full(len(df_1d), np.nan)
    weekly_s1 = np.full(len(df_1d), np.nan)
    
    for week in unique_weeks:
        week_mask = week_start == week
        if np.sum(week_mask) == 0:
            continue
        # Get Monday (first day of week)
        monday_idx = np.where(week_mask)[0][0]
        if monday_idx >= len(df_1d):
            continue
        # Use Monday's OHLC for weekly pivot
        monday_high = df_1d['high'].iloc[monday_idx]
        monday_low = df_1d['low'].iloc[monday_idx]
        monday_close = df_1d['close'].iloc[monday_idx]
        
        pivot = (monday_high + monday_low + monday_close) / 3
        range_val = monday_high - monday_low
        
        r1 = pivot + (range_val * 1.1 / 12)
        s1 = pivot - (range_val * 1.1 / 12)
        
        # Assign to all days in this week
        weekly_pivot[week_mask] = pivot
        weekly_r1[week_mask] = r1
        weekly_s1[week_mask] = s1
    
    # Calculate daily EMA(34) for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly pivot levels and daily EMA to 12h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume spike detection (24-period average on 12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Wait for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above weekly R1 with volume and in uptrend
            vol_condition = volume[i] > vol_ma[i] * 2.0
            uptrend = close[i] > ema_34_aligned[i]
            
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

# Hypothesis: 12h Weekly Pivot Breakout with Trend and Volume Confirmation
# Weekly pivot points calculated from Monday's OHLC provide significant weekly support/resistance.
# Breaks above R1 or below S1 with 2x volume indicate institutional interest.
# Daily EMA(34) ensures trades align with daily trend direction.
# Works in bull (buy R1 breaks in uptrend) and bear (sell S1 breaks in downtrend).
# Position size 0.25 balances risk and keeps trade frequency ~12-30/year on 12h timeframe.