#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_1d_Camarilla_R1S1_Breakout_Volume_Trend_v2"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily Close for Trend Filter ===
    close_1d = df_1d['close'].values
    sma200_1d = pd.Series(close_1d).rolling(window=200, min_periods=200).mean().values
    sma200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma200_1d)
    
    # === Daily 4h for Camarilla Pivot Calculation ===
    # Note: We use 4h data to calculate daily pivots (more granular than daily-only)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Group 4h bars into days (6 bars per day)
    # Calculate daily OHLC from 4h data
    n_4h = len(high_4h)
    days = n_4h // 6
    if days < 2:
        return np.zeros(n)
    
    # Reshape to get daily OHLC from 4h
    high_daily = np.max(high_4h[:days*6].reshape(days, 6), axis=1)
    low_daily = np.min(low_4h[:days*6].reshape(days, 6), axis=1)
    close_daily = close_4h[:days*6].reshape(days, 6)[:, -1]  # Last 4h bar of each day
    
    # Previous day's values for pivot calculation
    prev_high = np.roll(high_daily, 1)
    prev_low = np.roll(low_daily, 1)
    prev_close = np.roll(close_daily, 1)
    
    # Pivot point
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # Expand to 4h resolution (each day's levels apply to its 6 bars)
    r1_4h = np.repeat(r1, 6)
    s1_4h = np.repeat(s1, 6)
    pivot_4h = np.repeat(pivot, 6)
    
    # Align 4h data to 1h
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    pivot_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    
    # === 1h Data: Price, Volume, and Trend ===
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Volume ratio with proper initialization
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma20 > 0, volume / vol_ma20, 0)
    
    # 1h SMA50 trend filter
    close_series = pd.Series(close)
    sma50 = close_series.rolling(window=50, min_periods=50).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip outside session
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        sma50_val = sma50[i]
        sma200_1d_val = sma200_1d_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pivot_val = pivot_aligned[i]
        
        # Skip if any value is invalid
        if (np.isnan(vol_ratio_val) or np.isnan(sma50_val) or np.isnan(sma200_1d_val) or 
            np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(pivot_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation and uptrend (1h & daily)
            if (close_val > r1_val and 
                vol_ratio_val > 2.0 and 
                close_val > sma50_val and
                close_val > sma200_1d_val):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below S1 with volume confirmation and downtrend (1h & daily)
            elif (close_val < s1_val and 
                  vol_ratio_val > 2.0 and 
                  close_val < sma50_val and
                  close_val < sma200_1d_val):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below pivot or volume dries up
            if close_val < pivot_val or vol_ratio_val < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Price returns above pivot or volume dries up
            if close_val > pivot_val or vol_ratio_val < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals