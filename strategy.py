#!/usr/bin/env python3
"""
4h_WeeklyPivot_R1_S1_Breakout_Volume_Filter
Strategy: 4h Weekly Pivot R1/S1 breakout with volume confirmation and 12h trend filter.
Long: Price breaks above R1 + volume > 1.5x 12-period avg + 12h close > 12h open
Short: Price breaks below S1 + volume > 1.5x 12-period avg + 12h close < 12h open
Exit: Opposite pivot level touch or trend reversal
Position size: 0.25
Designed to capture breakouts in trending markets while avoiding false signals in ranging conditions.
Timeframe: 4h
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
    
    # Calculate Weekly Pivot levels for current week using previous week's OHLC
    def calculate_weekly_pivot(high, low, close):
        # Pivot point
        pp = (high + low + close) / 3
        # Range
        range_ = high - low
        # Weekly Pivot levels (R1, S1)
        r1 = pp + range_ * 1.1 / 12
        s1 = pp - range_ * 1.1 / 12
        return r1, s1
    
    # Need previous week's data to calculate this week's levels
    # We'll calculate for each bar using previous week's OHLC
    r1 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    
    # Convert to pandas for easier date handling
    df = prices.copy()
    df['date'] = pd.to_datetime(df['open_time']).dt.date
    df['week'] = pd.to_datetime(df['open_time']).dt.isocalendar().week
    df['year'] = pd.to_datetime(df['open_time']).dt.isocalendar().year
    
    # Group by year-week to get weekly OHLC
    weekly = df.groupby(['year', 'week']).agg({
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'open': 'first'
    }).reset_index()
    
    if len(weekly) < 2:
        return np.zeros(n)
    
    # Calculate Weekly Pivot levels for each week (using previous week's data)
    weekly['r1'] = np.nan
    weekly['s1'] = np.nan
    
    for i in range(1, len(weekly)):
        prev_high = weekly.iloc[i-1]['high']
        prev_low = weekly.iloc[i-1]['low']
        prev_close = weekly.iloc[i-1]['close']
        r1_val, s1_val = calculate_weekly_pivot(prev_high, prev_low, prev_close)
        weekly.iloc[i, weekly.columns.get_loc('r1')] = r1_val
        weekly.iloc[i, weekly.columns.get_loc('s1')] = s1_val
    
    # Map weekly levels back to 4h bars
    week_map = dict(zip(zip(weekly['year'], weekly['week']), zip(weekly['r1'], weekly['s1'])))
    for i in range(n):
        year_week = (df.iloc[i]['year'], df.iloc[i]['week'])
        if year_week in week_map:
            r1[i], s1[i] = week_map[year_week]
    
    # Calculate 12h trend (close > open = uptrend, close < open = downtrend)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    trend_12h = (df_12h['close'] > df_12h['open']).astype(float).values  # 1 for up, 0 for down
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Calculate 4h volume average (12-period)
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    volume_ma12_4h = pd.Series(volume_4h).rolling(window=12, min_periods=12).mean().values
    volume_ma12_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma12_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(50, n):  # warmup for indicators
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(trend_12h_aligned[i]) or 
            np.isnan(volume_ma12_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 4h volume
        vol_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)[i]
        volume_filter = vol_4h_current > (1.5 * volume_ma12_4h_aligned[i])
        
        # Trend filter: 12h bullish/bearish
        trend_up = trend_12h_aligned[i] > 0.5  # 12h close > open
        trend_down = trend_12h_aligned[i] < 0.5  # 12h close < open
        
        # Breakout conditions
        breakout_up = close[i] > r1[i]
        breakout_down = close[i] < s1[i]
        
        # Entry signals
        if position == 0:
            # Long: breakout above R1 + volume filter + 12h uptrend
            if breakout_up and volume_filter and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + volume filter + 12h downtrend
            elif breakout_down and volume_filter and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches S1 or 12h trend turns down
            if close[i] <= s1[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches R1 or 12h trend turns up
            if close[i] >= r1[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WeeklyPivot_R1_S1_Breakout_Volume_Filter"
timeframe = "4h"
leverage = 1.0