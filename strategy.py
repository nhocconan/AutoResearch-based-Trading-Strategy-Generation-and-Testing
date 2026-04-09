#!/usr/bin/env python3
# 6h_weekly_pivot_breakout_volume_v1
# Hypothesis: 6h strategy using weekly pivot points for structure, with breakout entries confirmed by volume and aligned with daily HTF EMA(50) trend. Weekly pivots provide significant support/resistance levels that price reacts to. Breakouts above weekly R1 or below weekly S1 with volume confirmation and daily trend alignment capture meaningful moves. Designed for low trade frequency (target: 12-37 trades/year) to minimize fee drag in both bull and bear markets. Uses discrete position sizing (0.25) to reduce churn.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Multi-timeframe: 1d EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    ema_50_1d = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Weekly pivot points (using prior week's OHLC)
    # We'll calculate weekly pivots from daily data and align to 6h
    df_1d_for_pivot = get_htf_data(prices, '1d')
    high_1d = df_1d_for_pivot['high'].values
    low_1d = df_1d_for_pivot['low'].values
    close_1d = df_1d_for_pivot['close'].values
    
    # Calculate weekly OHLC from daily data
    # Group daily data into weeks (starting Monday)
    dates_1d = pd.to_datetime(df_1d_for_pivot.index)
    week_num = dates_1d.isocalendar().week
    year_num = dates_1d.isocalendar().year
    
    # Create DataFrame for weekly aggregation
    weekly_df = pd.DataFrame({
        'high': high_1d,
        'low': low_1d,
        'close': close_1d,
        'week': week_num,
        'year': year_num
    }, index=dates_1d)
    
    # Group by year-week to get weekly OHLC
    weekly_ohlc = weekly_df.groupby(['year', 'week']).agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    weekly_ohlc['pivot'] = (weekly_ohlc['high'] + weekly_ohlc['low'] + weekly_ohlc['close']) / 3.0
    weekly_ohlc['r1'] = 2 * weekly_ohlc['pivot'] - weekly_ohlc['low']
    weekly_ohlc['s1'] = 2 * weekly_ohlc['pivot'] - weekly_ohlc['high']
    weekly_ohlc['r2'] = weekly_ohlc['pivot'] + (weekly_ohlc['high'] - weekly_ohlc['low'])
    weekly_ohlc['s2'] = weekly_ohlc['pivot'] - (weekly_ohlc['high'] - weekly_ohlc['low'])
    weekly_ohlc['r3'] = weekly_ohlc['high'] + 2 * (weekly_ohlc['pivot'] - weekly_ohlc['low'])
    weekly_ohlc['s3'] = weekly_ohlc['low'] - 2 * (weekly_ohlc['high'] - weekly_ohlc['pivot'])
    
    # Create mapping from date to weekly pivot values
    weekly_pivot_map = {}
    for _, row in weekly_ohlc.iterrows():
        week_start = pd.to_datetime(f"{row['year']}-W{int(row['week'])-1}", format='%Y-W%W-%w')
        weekly_pivot_map[week_start] = {
            'pivot': row['pivot'],
            'r1': row['r1'],
            's1': row['s1'],
            'r2': row['r2'],
            's2': row['s2'],
            'r3': row['r3'],
            's3': row['s3']
        }
    
    # For each 6h bar, find the most recent weekly pivot values
    # We'll use the prior week's completed pivots to avoid look-ahead
    pivot_values = np.full(n, np.nan)
    r1_values = np.full(n, np.nan)
    s1_values = np.full(n, np.nan)
    r2_values = np.full(n, np.nan)
    s2_values = np.full(n, np.nan)
    r3_values = np.full(n, np.nan)
    s3_values = np.full(n, np.nan)
    
    # Get 6h timestamps
    prices_index = prices.index
    
    for i in range(n):
        current_time = prices_index[i]
        # Find the most recent Monday (start of week) prior to current bar
        # Subtract days to get to previous Monday, then use that week's pivots
        days_since_monday = current_time.weekday()  # Monday=0, Sunday=6
        monday_this_week = current_time - pd.Timedelta(days=days_since_monday)
        monday_prior_week = monday_this_week - pd.Timedelta(weeks=1)
        
        # Use prior week's pivots (completed week)
        if monday_prior_week in weekly_pivot_map:
            pivots = weekly_pivot_map[monday_prior_week]
            pivot_values[i] = pivots['pivot']
            r1_values[i] = pivots['r1']
            s1_values[i] = pivots['s1']
            r2_values[i] = pivots['r2']
            s2_values[i] = pivots['s2']
            r3_values[i] = pivots['r3']
            s3_values[i] = pivots['s3']
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(pivot_values[i]) or np.isnan(r1_values[i]) or np.isnan(s1_values[i]) or
            np.isnan(r2_values[i]) or np.isnan(s2_values[i]) or np.isnan(r3_values[i]) or np.isnan(s3_values[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # HTF trend filter: price above/below 1d EMA(50)
        htf_uptrend = close[i] > ema_50_1d_aligned[i]
        htf_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below weekly pivot (mean reversion to mean)
            if close[i] < pivot_values[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly pivot (mean reversion to mean)
            if close[i] > pivot_values[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout above weekly R1 with volume and HTF uptrend
            bullish_breakout = (close[i] > r1_values[i-1]) and volume_confirmed and htf_uptrend
            # Check for breakdown below weekly S1 with volume and HTF downtrend
            bearish_breakout = (close[i] < s1_values[i-1]) and volume_confirmed and htf_downtrend
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals