#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_DailyPivot_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 1:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard)
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # Standard pivot: P = (H + L + C)/3
    pivot_d = (high_d + low_d + close_d) / 3.0
    # Resistance/Support levels
    r1_d = 2 * pivot_d - low_d
    s1_d = 2 * pivot_d - high_d
    r2_d = pivot_d + (high_d - low_d)
    s2_d = pivot_d - (high_d - low_d)
    
    # Align daily pivot levels to 4h timeframe
    pivot_d_aligned = align_htf_to_ltf(prices, df_d, pivot_d)
    r1_d_aligned = align_htf_to_ltf(prices, df_d, r1_d)
    s1_d_aligned = align_htf_to_ltf(prices, df_d, s1_d)
    r2_d_aligned = align_htf_to_ltf(prices, df_d, r2_d)
    s2_d_aligned = align_htf_to_ltf(prices, df_d, s2_d)
    
    # Get weekly data for trend filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on weekly close
    close_w = df_w['close'].values
    ema_w = pd.Series(close_w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_w_aligned = align_htf_to_ltf(prices, df_w, ema_w)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need enough data for weekly EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(pivot_d_aligned[i]) or 
            np.isnan(r1_d_aligned[i]) or
            np.isnan(s1_d_aligned[i]) or
            np.isnan(r2_d_aligned[i]) or
            np.isnan(s2_d_aligned[i]) or
            np.isnan(ema_w_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot_val = pivot_d_aligned[i]
        r1_val = r1_d_aligned[i]
        s1_val = s1_d_aligned[i]
        r2_val = r2_d_aligned[i]
        s2_val = s2_d_aligned[i]
        ema_w_val = ema_w_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: Price above R1 + above weekly EMA + volume filter
            if close[i] > r1_val and close[i] > ema_w_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Price below S1 + below weekly EMA + volume filter
            elif close[i] < s1_val and close[i] < ema_w_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below S1 or below weekly EMA
            if close[i] < s1_val or close[i] < ema_w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above R1 or above weekly EMA
            if close[i] > r1_val or close[i] > ema_w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals