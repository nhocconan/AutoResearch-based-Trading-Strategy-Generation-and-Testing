#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_WeeklyPivot_R1S1_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot from previous week (7 days)
    # Shift by 7 to get previous week's data
    prev_week_high = np.roll(high_1d, 7)
    prev_week_low = np.roll(low_1d, 7)
    prev_week_close = np.roll(close_1d, 7)
    prev_week_high[:7] = np.nan
    prev_week_low[:7] = np.nan
    prev_week_close[:7] = np.nan
    
    # Weekly pivot = (H + L + C) / 3
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    # Weekly R1 = C + (H - L) * 1.1 / 12
    weekly_r1 = prev_week_close + (prev_week_high - prev_week_low) * 1.1 / 12.0
    # Weekly S1 = C - (H - L) * 1.1 / 12
    weekly_s1 = prev_week_close - (prev_week_high - prev_week_low) * 1.1 / 12.0
    
    # Align to 6h timeframe
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_6h = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_6h = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # Time filter: 00-23 UTC (full day)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    time_filter = (hours >= 0) & (hours <= 23)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if not time_filter[i]:
            signals[i] = 0.0
            continue
            
        if np.isnan(weekly_pivot_6h[i]) or np.isnan(weekly_r1_6h[i]) or np.isnan(weekly_s1_6h[i]) or \
           np.isnan(vol_ma_30[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_30[i]
        
        # Volume filter: current volume > 1.8x average
        volume_filter = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long: Price breaks above weekly R1 with volume filter
            if price > weekly_r1_6h[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 with volume filter
            elif price < weekly_s1_6h[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below weekly pivot (reversal signal)
            if price < weekly_pivot_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above weekly pivot (reversal signal)
            if price > weekly_pivot_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals