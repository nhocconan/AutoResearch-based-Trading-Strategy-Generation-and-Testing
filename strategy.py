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
    
    # Load daily data for 100-period EMA and weekly pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily EMA100 for trend
    ema_100_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 100:
        ema_100_1d[99] = np.mean(close_1d[:100])
        for i in range(100, len(close_1d)):
            ema_100_1d[i] = (close_1d[i] * 2 + ema_100_1d[i-1] * 98) / 100
    
    # Calculate weekly pivot points from daily data (prior week)
    # We need to group by week to get proper weekly OHLC
    # For simplicity, we'll use daily OHLC from 7 days prior as proxy for weekly
    weekly_pivot = np.full_like(close_1d, np.nan)
    weekly_r1 = np.full_like(close_1d, np.nan)
    weekly_s1 = np.full_like(close_1d, np.nan)
    weekly_r2 = np.full_like(close_1d, np.nan)
    weekly_s2 = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 8:  # Need at least a week of data
        for i in range(7, len(close_1d)):
            # Weekly OHLC from 7 days ago to yesterday
            week_high = np.max(high_1d[i-7:i])  # High of past 7 days
            week_low = np.min(low_1d[i-7:i])    # Low of past 7 days
            week_close = close_1d[i-1]          # Close of yesterday
            
            pp = (week_high + week_low + week_close) / 3.0
            r1 = 2 * pp - week_low
            s1 = 2 * pp - week_high
            r2 = pp + (week_high - week_low)
            s2 = pp - (week_high - week_low)
            
            weekly_pivot[i] = pp
            weekly_r1[i] = r1
            weekly_s1[i] = s1
            weekly_r2[i] = r2
            weekly_s2[i] = s2
    
    # Align daily indicators to 6h timeframe
    ema_100_1d_6h = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_6h = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_6h = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_r2_6h = align_htf_to_ltf(prices, df_1d, weekly_r2)
    weekly_s2_6h = align_htf_to_ltf(prices, df_1d, weekly_s2)
    
    # Volume confirmation: 20-period volume average
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_100_1d_6h[i]) or 
            np.isnan(weekly_pivot_6h[i]) or 
            np.isnan(weekly_r1_6h[i]) or 
            np.isnan(weekly_s1_6h[i]) or
            np.isnan(weekly_r2_6h[i]) or 
            np.isnan(weekly_s2_6h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Long: Price crosses above weekly S2 with volume and above EMA100
            if (close[i] > weekly_s2_6h[i] and
                close[i] > ema_100_1d_6h[i] and
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: Price crosses below weekly R2 with volume and below EMA100
            elif (close[i] < weekly_r2_6h[i] and
                  close[i] < ema_100_1d_6h[i] and
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price crosses below weekly S1 or below EMA100
            if (close[i] < weekly_s1_6h[i] or 
                close[i] < ema_100_1d_6h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price crosses above weekly R1 or above EMA100
            if (close[i] > weekly_r1_6h[i] or 
                close[i] > ema_100_1d_6h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_WeeklyPivot_S2R2_EMA100_Volume"
timeframe = "6h"
leverage = 1.0