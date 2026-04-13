#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot levels from daily data
    # Weekly high/low/close from last completed week
    weekly_high = np.full(len(close_1d), np.nan)
    weekly_low = np.full(len(close_1d), np.nan)
    weekly_close = np.full(len(close_1d), np.nan)
    
    # Simple approach: use rolling window of 5 days for weekly approximation
    for i in range(len(close_1d)):
        if i >= 4:
            weekly_high[i] = np.max(high_1d[i-4:i+1])
            weekly_low[i] = np.min(low_1d[i-4:i+1])
            weekly_close[i] = close_1d[i]
    
    # Calculate pivot points and support/resistance levels
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    r4 = r3 + (weekly_high - weekly_low)
    s4 = s3 - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume above 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(pivot_6h[i]) or np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(r4_6h[i]) or 
            np.isnan(s4_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions with volume confirmation
        long_breakout = (close[i] > r4_6h[i]) and vol_surge[i]
        short_breakout = (close[i] < s4_6h[i]) and vol_surge[i]
        
        # Fade conditions at extreme levels
        long_fade = (close[i] < s3_6h[i]) and vol_surge[i]
        short_fade = (close[i] > r3_6h[i]) and vol_surge[i]
        
        # Exit conditions: return to pivot or opposite signal
        exit_long = position == 1 and (close[i] < pivot_6h[i] or close[i] > r4_6h[i])
        exit_short = position == -1 and (close[i] > pivot_6h[i] or close[i] < s4_6h[i])
        
        # Execute signals
        if (long_breakout or long_fade) and position != 1:
            position = 1
            signals[i] = position_size
        elif (short_breakout or short_fade) and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_weekly_pivot_breakout_fade"
timeframe = "6h"
leverage = 1.0