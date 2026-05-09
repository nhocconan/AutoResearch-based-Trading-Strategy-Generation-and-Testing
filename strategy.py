#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate EMA50 on weekly close for trend filter
    close_weekly = df_weekly['close'].values
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Calculate weekly pivot levels from previous weekly bar
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly_vals = df_weekly['close'].values
    
    # Weekly pivot point and key levels
    pivot_point = (high_weekly + low_weekly + close_weekly_vals) / 3
    weekly_range = high_weekly - low_weekly
    
    # Weekly resistance and support levels (similar to Camarilla but using weekly range)
    r1 = pivot_point + (weekly_range * 1.1 / 2)
    s1 = pivot_point - (weekly_range * 1.1 / 2)
    r2 = pivot_point + weekly_range
    s2 = pivot_point - weekly_range
    
    # Align weekly pivot levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # Volume spike filter: current volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema50_weekly_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50 = ema50_weekly_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        r2 = r2_aligned[i]
        s2 = s2_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Close breaks above R2 + weekly uptrend + volume spike
            if close[i] > r2 and close[i] > ema50 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Close breaks below S2 + weekly downtrend + volume spike
            elif close[i] < s2 and close[i] < ema50 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close falls below S1 or weekly trend turns down
            if close[i] < s1 or close[i] < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close rises above R1 or weekly trend turns up
            if close[i] > r1 or close[i] > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals