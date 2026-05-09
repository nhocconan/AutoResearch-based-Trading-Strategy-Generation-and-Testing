#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_DailyBreakout_TrendFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    
    # Align weekly pivots to daily
    pivot_d = align_htf_to_ltf(prices, df_1w, pivot)
    r1_d = align_htf_to_ltf(prices, df_1w, r1)
    s1_d = align_htf_to_ltf(prices, df_1w, s1)
    r2_d = align_htf_to_ltf(prices, df_1w, r2)
    s2_d = align_htf_to_ltf(prices, df_1w, s2)
    
    # Weekly trend filter: price above/below weekly EMA20
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema20_d = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    
    # Daily volume filter: volume above 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for weekly EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_d[i]) or np.isnan(r1_d[i]) or np.isnan(s1_d[i]) or
            np.isnan(r2_d[i]) or np.isnan(s2_d[i]) or np.isnan(weekly_ema20_d[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        # Pre-compute hour for session filter (UTC 8-20)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long: Price breaks above R1 with volume, in uptrend, during session
            if (close[i] > r1_d[i] and 
                close[i] > weekly_ema20_d[i] and  # weekly uptrend filter
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume, in downtrend, during session
            elif (close[i] < s1_d[i] and 
                  close[i] < weekly_ema20_d[i] and  # weekly downtrend filter
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below weekly pivot or trend changes
            if close[i] < pivot_d[i] or close[i] < weekly_ema20_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above weekly pivot or trend changes
            if close[i] > pivot_d[i] or close[i] > weekly_ema20_d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals