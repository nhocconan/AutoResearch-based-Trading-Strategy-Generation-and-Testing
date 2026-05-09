#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_DailyTrend_VolumeFilter"
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
    
    # Get weekly data for pivot points (use previous week's data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's H/L/C)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot levels for each week
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily EMA to 6h
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume confirmation (20-period average)
    vol_avg_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            vol_avg_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_avg_today = vol_avg_20[i]
        vol_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = vol_current > 1.5 * vol_avg_today
        
        # Get weekly pivot levels for current bar
        pivot_level = pivot_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        r2_level = r2_aligned[i]
        s2_level = s2_aligned[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        
        # Get daily trend
        ema34_today = ema34_1d_aligned[i]
        
        if position == 0:
            # Long conditions: price near support with bullish trend and volume
            # Look for bounce off S1, S2, or S3 with volume confirmation
            near_support = (abs(price - s1_level) / s1_level < 0.02 or 
                           abs(price - s2_level) / s2_level < 0.025 or
                           abs(price - s3_level) / s3_level < 0.03)
            bullish_trend = price > ema34_today
            
            if near_support and bullish_trend and vol_confirmed:
                signals[i] = 0.25
                position = 1
                continue
            
            # Short conditions: price near resistance with bearish trend and volume
            # Look for rejection at R1, R2, or R3 with volume confirmation
            near_resistance = (abs(price - r1_level) / r1_level < 0.02 or 
                              abs(price - r2_level) / r2_level < 0.025 or
                              abs(price - r3_level) / r3_level < 0.03)
            bearish_trend = price < ema34_today
            
            if near_resistance and bearish_trend and vol_confirmed:
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price breaks below pivot or trend changes
            exit_signal = False
            if price < pivot_level:  # Break below pivot
                exit_signal = True
            elif price < ema34_today:  # Trend change to bearish
                exit_signal = True
            elif not vol_confirmed:  # Volume confirmation lost
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above pivot or trend changes
            exit_signal = False
            if price > pivot_level:  # Break above pivot
                exit_signal = True
            elif price > ema34_today:  # Trend change to bullish
                exit_signal = True
            elif not vol_confirmed:  # Volume confirmation lost
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals