#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for weekly pivot points (based on weekly OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points using weekly OHLC
    # We'll compute weekly values from daily data
    weeks = len(df_1d) // 7
    weekly_high = np.zeros(weeks)
    weekly_low = np.zeros(weeks)
    weekly_close = np.zeros(weeks)
    
    for w in range(weeks):
        start_idx = w * 7
        end_idx = start_idx + 7
        if end_idx <= len(df_1d):
            weekly_high[w] = np.max(high_1d[start_idx:end_idx])
            weekly_low[w] = np.min(low_1d[start_idx:end_idx])
            weekly_close[w] = close_1d[end_idx - 1]
    
    # Calculate pivot points for each week
    weekly_pivot = np.zeros(weeks)
    weekly_r1 = np.zeros(weeks)
    weekly_r2 = np.zeros(weeks)
    weekly_s1 = np.zeros(weeks)
    weekly_s2 = np.zeros(weeks)
    
    for w in range(weeks):
        ph = weekly_high[w]
        pl = weekly_low[w]
        pc = weekly_close[w]
        
        pp = (ph + pl + pc) / 3.0
        r1 = 2 * pp - pl
        r2 = pp + (ph - pl)
        s1 = 2 * pp - ph
        s2 = pp - (ph - pl)
        
        weekly_pivot[w] = pp
        weekly_r1[w] = r1
        weekly_r2[w] = r2
        weekly_s1[w] = s1
        weekly_s2[w] = s2
    
    # Expand weekly values to daily resolution (each value applies to 7 days)
    pivot_daily = np.repeat(weekly_pivot, 7)
    r1_daily = np.repeat(weekly_r1, 7)
    r2_daily = np.repeat(weekly_r2, 7)
    s1_daily = np.repeat(weekly_s1, 7)
    s2_daily = np.repeat(weekly_s2, 7)
    
    # Trim to match daily data length
    pivot_daily = pivot_daily[:len(df_1d)]
    r1_daily = r1_daily[:len(df_1d)]
    r2_daily = r2_daily[:len(df_1d)]
    s1_daily = s1_daily[:len(df_1d)]
    s2_daily = s2_daily[:len(df_1d)]
    
    # Align weekly pivots to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot_daily)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1_daily)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2_daily)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1_daily)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2_daily)
    
    # Volume spike detection (24-period average on 6h = 4 days)
    vol_ma_24 = np.full_like(volume, np.nan)
    if len(volume) >= 24:
        for i in range(23, len(volume)):
            vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_6h[i]) or 
            np.isnan(r1_6h[i]) or
            np.isnan(r2_6h[i]) or
            np.isnan(s1_6h[i]) or
            np.isnan(s2_6h[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 24-period average
        if vol_ma_24[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_24[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.5
        
        if position == 0:
            # Long: Price breaks above R1 with volume spike
            if (close[i] > r1_6h[i] and 
                volume_ratio > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below S1 with volume spike
            elif (close[i] < s1_6h[i] and 
                  volume_ratio > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price closes below weekly pivot (mean reversion to weekly mean)
            if close[i] < pivot_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price closes above weekly pivot (mean reversion to weekly mean)
            if close[i] > pivot_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WeeklyPivot_Volume_Breakout"
timeframe = "6h"
leverage = 1.0