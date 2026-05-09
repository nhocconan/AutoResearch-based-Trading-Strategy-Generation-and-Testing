#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_DailyTrend_Volume"
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
    
    # Get daily data for trend filter and weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation (weekly high/low/close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Previous week's OHLC for weekly pivot (to avoid look-ahead)
    prev_weekly_close = df_1w['close'].shift(1).values
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    
    # Calculate weekly pivot point and key levels
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - prev_weekly_low  # Resistance 1
    weekly_s1 = 2 * weekly_pivot - prev_weekly_high  # Support 1
    weekly_r2 = weekly_pivot + (prev_weekly_high - prev_weekly_low)  # Resistance 2
    weekly_s2 = weekly_pivot - (prev_weekly_high - prev_weekly_low)  # Support 2
    
    # Align weekly pivot levels to 6h
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_6h = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_6h = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r2_6h = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_6h = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: above 1.5x 4-period average (4*6h = 1 day)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 4  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_6h[i]) or np.isnan(weekly_r1_6h[i]) or 
            np.isnan(weekly_s1_6h[i]) or np.isnan(weekly_r2_6h[i]) or 
            np.isnan(weekly_s2_6h[i]) or np.isnan(ema_34_6h[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]  # Volume confirmation
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long entry: price above weekly pivot with daily uptrend and volume
            if (close[i] > weekly_pivot_6h[i] and 
                close[i] > ema_34_6h[i] and  # daily uptrend
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short entry: price below weekly pivot with daily downtrend and volume
            elif (close[i] < weekly_pivot_6h[i] and 
                  close[i] < ema_34_6h[i] and  # daily downtrend
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly support 1
            if not np.isnan(weekly_s1_6h[i]) and close[i] < weekly_s1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly resistance 1
            if not np.isnan(weekly_r1_6h[i]) and close[i] > weekly_r1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals