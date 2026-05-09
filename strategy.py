#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for trend filter and weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate weekly pivot from previous week (using daily data)
    # Need at least 5 days for weekly pivot calculation
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Get previous week's high, low, close (excluding current day)
    week_high = df_1d['high'].iloc[-6:-1].max() if len(df_1d) >= 6 else df_1d['high'].iloc[:-1].max()
    week_low = df_1d['low'].iloc[-6:-1].min() if len(df_1d) >= 6 else df_1d['low'].iloc[:-1].min()
    week_close = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[0]
    
    # Calculate weekly pivot levels
    pivot_point = (week_high + week_low + week_close) / 3.0
    r1 = 2 * pivot_point - week_low
    s1 = 2 * pivot_point - week_high
    r2 = pivot_point + (week_high - week_low)
    s2 = pivot_point - (week_high - week_low)
    r3 = week_high + 2 * (pivot_point - week_low)
    s3 = week_low - 2 * (week_high - pivot_point)
    
    # Broadcast weekly pivot levels to all 1d bars (they change only weekly)
    pp_array = np.full_like(close_1d, pivot_point)
    r1_array = np.full_like(close_1d, r1)
    s1_array = np.full_like(close_1d, s1)
    r2_array = np.full_like(close_1d, r2)
    s2_array = np.full_like(close_1d, s2)
    r3_array = np.full_like(close_1d, r3)
    s3_array = np.full_like(close_1d, s3)
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_array)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_array)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_array)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_array)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_array)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_array)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_array)
    
    # Volume spike filter: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1d = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Close > R3 and price above 1d EMA34 with volume spike
            if close[i] > r3_aligned[i] and close[i] > ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Close < S3 and price below 1d EMA34 with volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close < S1 or trend breaks (price < 1d EMA34)
            if close[i] < s1_aligned[i] or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close > R1 or trend breaks (price > 1d EMA34)
            if close[i] > r1_aligned[i] or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals