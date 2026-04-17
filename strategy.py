#!/usr/bin/env python3
"""
12h_WeeklyPivot_R1_S1_Breakout_VolumeFilter_v4
Strategy: 12h Weekly pivot R1/S1 breakout with volume confirmation and 1w EMA50 trend filter.
Long: Price breaks above R1 + volume > 1.3x 20-period avg + price > weekly EMA50
Short: Price breaks below S1 + volume > 1.3x 20-period avg + price < weekly EMA50
Exit: Opposite pivot level touch or trend reversal
Position size: 0.25
Designed to capture breakouts in trending markets while avoiding false signals in ranging conditions.
Timeframe: 12h
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
    volume = prices['volume'].values
    
    # Calculate Weekly pivot levels for current week using previous week's OHLC
    def calculate_pivot(high, low, close):
        # Pivot point
        pp = (high + low + close) / 3
        # Range
        range_ = high - low
        # Weekly R1/S1 (similar to Camarilla but using 1/6 for sensitivity)
        r1 = pp + range_ * 1.0 / 6
        s1 = pp - range_ * 1.0 / 6
        return r1, s1
    
    # Need previous week's data to calculate current week's levels
    # We'll calculate for each bar using previous week's OHLC
    r1 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    
    # Convert to pandas for easier date handling
    df = prices.copy()
    df['date'] = pd.to_datetime(df['open_time']).dt.date
    
    # Group by week to get weekly OHLC (Monday to Sunday)
    df['week_start'] = df['date'] - pd.to_timedelta(df['date'].dt.weekday, unit='D')
    weekly = df.groupby('week_start').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(weekly) < 2:
        return np.zeros(n)
    
    # Calculate Weekly pivot levels for each week (using previous week's data)
    weekly['r1'] = np.nan
    weekly['s1'] = np.nan
    
    for i in range(1, len(weekly)):
        prev_high = weekly.iloc[i-1]['high']
        prev_low = weekly.iloc[i-1]['low']
        prev_close = weekly.iloc[i-1]['close']
        r1_val, s1_val = calculate_pivot(prev_high, prev_low, prev_close)
        weekly.iloc[i, weekly.columns.get_loc('r1')] = r1_val
        weekly.iloc[i, weekly.columns.get_loc('s1')] = s1_val
    
    # Map weekly levels back to 12h bars
    week_map = dict(zip(weekly['week_start'], zip(weekly['r1'], weekly['s1'])))
    for i in range(n):
        week_start = pd.to_datetime(df.iloc[i]['week_start']).date()
        if week_start in week_map:
            r1[i], s1[i] = week_map[week_start]
    
    # Calculate weekly EMA50 for trend filter (using 1w data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h volume average (20-period)
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    volume_ma20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(60, n):  # warmup for indicators
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_ma20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h volume
        vol_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)[i]
        volume_filter = vol_12h_current > (1.3 * volume_ma20_12h_aligned[i])
        
        # Trend filter: price above/below weekly EMA50
        trend_up = close[i] > ema_50_1w_aligned[i]
        trend_down = close[i] < ema_50_1w_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > r1[i]
        breakout_down = close[i] < s1[i]
        
        # Entry signals
        if position == 0:
            # Long: breakout above R1 + volume filter + trend up
            if breakout_up and volume_filter and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + volume filter + trend down
            elif breakout_down and volume_filter and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches S1 or trend down
            if close[i] <= s1[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches R1 or trend up
            if close[i] >= r1[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyPivot_R1_S1_Breakout_VolumeFilter_v4"
timeframe = "12h"
leverage = 1.0