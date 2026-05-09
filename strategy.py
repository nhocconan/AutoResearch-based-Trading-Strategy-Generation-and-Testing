#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_TrendFollowing_VolumeFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot Point = (H + L + C) / 3
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    # Support 1 = (2 * PP) - H
    s1 = (2 * pp) - weekly_high
    # Resistance 1 = (2 * PP) - L
    r1 = (2 * pp) - weekly_low
    # Support 2 = PP - (H - L)
    s2 = pp - (weekly_high - weekly_low)
    # Resistance 2 = PP + (H - L)
    r2 = pp + (weekly_high - weekly_low)
    
    # Align weekly pivots to daily timeframe (with 1-bar delay for completed weekly bar)
    pp_d = align_htf_to_ltf(prices, df_1w, pp)
    s1_d = align_htf_to_ltf(prices, df_1w, s1)
    r1_d = align_htf_to_ltf(prices, df_1w, r1)
    s2_d = align_htf_to_ltf(prices, df_1w, s2)
    r2_d = align_htf_to_ltf(prices, df_1w, r2)
    
    # Volume filter: spike above 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(pp_d[i]) or np.isnan(s1_d[i]) or np.isnan(r1_d[i]) or 
            np.isnan(s2_d[i]) or np.isnan(r2_d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]  # Volume confirmation
        
        # Pre-compute hour for session filter (UTC 0-24)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Trade during active hours (8 AM - 8 PM UTC) to avoid low-volume periods
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: price above weekly S1, volume breakout
            if (close[i] > s1_d[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly R1, volume breakdown
            elif (close[i] < r1_d[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below weekly S2
            if close[i] < s2_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above weekly R2
            if close[i] > r2_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals