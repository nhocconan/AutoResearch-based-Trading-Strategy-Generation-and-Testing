#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_R3S4_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for weekly pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate weekly pivot points from daily data
    # Group daily data into weeks (starting Monday)
    df_1d_copy = df_1d.copy()
    df_1d_copy['week_start'] = df_1d_copy.index.to_series().dt.to_period('W').apply(lambda r: r.start_time)
    weekly = df_1d_copy.groupby('week_start').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot and ranges (using previous week)
    weekly['prev_high'] = weekly['high'].shift(1)
    weekly['prev_low'] = weekly['low'].shift(1)
    weekly['prev_close'] = weekly['close'].shift(1)
    
    weekly['pivot'] = (weekly['prev_high'] + weekly['prev_low'] + weekly['prev_close']) / 3
    weekly['range'] = weekly['prev_high'] - weekly['prev_low']
    
    # Weekly R3 and S4 levels
    weekly['r3'] = weekly['prev_close'] + weekly['range'] * 1.1 / 2
    weekly['s4'] = weekly['prev_close'] - weekly['range'] * 1.1 * 2  # S4 = close - 2.2 * range
    
    # Forward fill weekly values to daily index
    weekly_index = pd.Index(weekly['week_start'])
    daily_to_weekly = df_1d_copy.index.to_series().dt.to_period('W').apply(lambda r: r.start_time)
    
    r3_weekly = weekly.set_index('week_start')['r3'].reindex(daily_to_weekly).values
    s4_weekly = weekly.set_index('week_start')['s4'].reindex(daily_to_weekly).values
    
    # Align weekly levels to 6h timeframe
    r3_weekly_aligned = align_htf_to_ltf(prices, df_1d, r3_weekly)
    s4_weekly_aligned = align_htf_to_ltf(prices, df_1d, s4_weekly)
    
    # Volume spike detection: current volume > 2.0 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for daily and weekly calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(r3_weekly_aligned[i]) or 
            np.isnan(s4_weekly_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema34_1d_aligned[i]
        r3_val = r3_weekly_aligned[i]
        s4_val = s4_weekly_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above R3 with volume spike, above 1d EMA
            if (close[i] > r3_val and vol_spike and 
                close[i] > ema_val):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S4 with volume spike, below 1d EMA
            elif (close[i] < s4_val and vol_spike and 
                  close[i] < ema_val):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S4 OR below 1d EMA
            if (close[i] < s4_val or close[i] < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 OR above 1d EMA
            if (close[i] > r3_val or close[i] > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals