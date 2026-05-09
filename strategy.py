#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_Breakout_Momentum_v2"
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
    
    # Get weekly data for pivot calculation
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Standard pivot: P = (H + L + C)/3
    pivot_w = (high_w + low_w + close_w) / 3.0
    # Resistance/Support levels
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    
    # Align weekly pivot levels to daily timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_w, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_w, s2_w)
    
    # Get daily data for momentum filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 30:
        return np.zeros(n)
    
    # Calculate 30-period EMA on daily close for trend filter
    close_d = df_d['close'].values
    ema_30_d = pd.Series(close_d).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_d_aligned = align_htf_to_ltf(prices, df_d, ema_30_d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Need enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(pivot_w_aligned[i]) or 
            np.isnan(r1_w_aligned[i]) or
            np.isnan(s1_w_aligned[i]) or
            np.isnan(r2_w_aligned[i]) or
            np.isnan(s2_w_aligned[i]) or
            np.isnan(ema_30_d_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot_w_val = pivot_w_aligned[i]
        r1_w_val = r1_w_aligned[i]
        s1_w_val = s1_w_aligned[i]
        r2_w_val = r2_w_aligned[i]
        s2_w_val = s2_w_aligned[i]
        ema_30_val = ema_30_d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: Price above R1 + price above weekly EMA30 + volume filter
            if close[i] > r1_w_val and close[i] > ema_30_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Price below S1 + price below weekly EMA30 + volume filter
            elif close[i] < s1_w_val and close[i] < ema_30_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below S1 or below weekly EMA30
            if close[i] < s1_w_val or close[i] < ema_30_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above R1 or above weekly EMA30
            if close[i] > r1_w_val or close[i] > ema_30_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals