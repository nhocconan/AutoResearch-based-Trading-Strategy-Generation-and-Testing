#!/usr/bin/env python3
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
    
    # Get daily data for weekly pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 3:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week's data
    # Group by week (Monday to Sunday)
    df_1d_copy = df_1d.copy()
    df_1d_copy['week'] = pd.to_datetime(df_1d_copy.index).isocalendar().week
    df_1d_copy['year'] = pd.to_datetime(df_1d_copy.index).isocalendar().year
    
    # Calculate weekly high, low, close
    weekly_data = df_1d_copy.groupby(['year', 'week']).agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(weekly_data) < 2:
        return np.zeros(n)
    
    # Calculate pivot points: P = (H + L + C)/3
    weekly_data['pivot'] = (weekly_data['high'] + weekly_data['low'] + weekly_data['close']) / 3
    # Resistance levels: R1 = 2*P - L, R2 = P + (H - L), R3 = H + 2*(P - L)
    weekly_data['r1'] = 2 * weekly_data['pivot'] - weekly_data['low']
    weekly_data['r2'] = weekly_data['pivot'] + (weekly_data['high'] - weekly_data['low'])
    weekly_data['r3'] = weekly_data['high'] + 2 * (weekly_data['pivot'] - weekly_data['low'])
    # Support levels: S1 = 2*P - H, S2 = P - (H - L), S3 = L - 2*(H - P)
    weekly_data['s1'] = 2 * weekly_data['pivot'] - weekly_data['high']
    weekly_data['s2'] = weekly_data['pivot'] - (weekly_data['high'] - weekly_data['low'])
    weekly_data['s3'] = weekly_data['low'] - 2 * (weekly_data['high'] - weekly_data['pivot'])
    
    # Create arrays for each level
    weeks = weekly_data[['year', 'week']].values
    pivot_vals = weekly_data['pivot'].values
    r3_vals = weekly_data['r3'].values
    s3_vals = weekly_data['s3'].values
    
    # Map each daily bar to its corresponding week's pivot levels
    pivot_daily = np.full(len(df_1d), np.nan)
    r3_daily = np.full(len(df_1d), np.nan)
    s3_daily = np.full(len(df_1d), np.nan)
    
    # For each daily bar, find the week it belongs to and assign that week's levels
    for i in range(len(df_1d)):
        date = df_1d.index[i]
        year = date.isocalendar().year
        week = date.isocalendar().week
        # Find matching week in weekly_data
        match_idx = np.where((weeks[:, 0] == year) & (weeks[:, 1] == week))[0]
        if len(match_idx) > 0:
            idx = match_idx[0]
            pivot_daily[i] = pivot_vals[idx]
            r3_daily[i] = r3_vals[idx]
            s3_daily[i] = s3_vals[idx]
    
    # Align to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot_daily)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3_daily)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3_daily)
    
    # Calculate 60-period EMA for trend filter (using 6h data)
    close_s = pd.Series(close)
    ema_60 = close_s.ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Volume filter: 6h volume > 1.5x 20-period average
    volume_s = pd.Series(volume)
    vol_ma_20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup
    start_idx = max(60, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(ema_60[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        pivot = pivot_6h[i]
        r3 = r3_6h[i]
        s3 = s3_6h[i]
        ema = ema_60[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions
        if position == 0:
            # Long: price above R3 + above EMA + volume
            if close[i] > r3 and close[i] > ema and vol_filter:
                signals[i] = size
                position = 1
            # Short: price below S3 + below EMA + volume
            elif close[i] < s3 and close[i] < ema and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price below pivot or trend change
            if close[i] < pivot or close[i] < ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price above pivot or trend change
            if close[i] > pivot or close[i] > ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_R3S3_Breakout_EMA60_VolumeFilter"
timeframe = "6h"
leverage = 1.0