#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w weekly pivot alignment
    # Weekly pivot from previous 5 trading days provides institutional bias
    # Volume > 1.5x 20-period average confirms breakout validity
    # Target: 30-100 total trades over 4 years (7-25/year) for low fee drag
    # Works in both bull and bear via directional pivot bias and breakout confirmation
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous week's pivot (using previous 5 trading days ~ 1 week)
    # Weekly high/low/close from 5d ago to 1d ago
    weekly_high = np.full(len(high_1d), np.nan)
    weekly_low = np.full(len(low_1d), np.nan)
    weekly_close = np.full(len(close_1d), np.nan)
    
    for i in range(5, len(high_1d)):
        weekly_high[i] = np.max(high_1d[i-5:i])   # Previous 5 days high
        weekly_low[i] = np.min(low_1d[i-5:i])     # Previous 5 days low
        weekly_close[i] = close_1d[i-1]           # Previous day close
    
    # Weekly pivot points (standard calculation)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    
    # Get 1d Donchian(20) for breakout
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Get 1d volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    # Align all indicators to LTF (1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Weekly pivot bias
        bullish_bias = close[i] > pivot_aligned[i]
        bearish_bias = close[i] < pivot_aligned[i]
        
        # Strong bullish/bearish conditions (beyond R1/S1)
        strong_bullish = close[i] > r1_aligned[i]
        strong_bearish = close[i] < s1_aligned[i]
        
        # Entry logic: Breakout + pivot alignment + volume confirmation
        long_entry = long_breakout and bullish_bias and strong_bullish and volume_spike[i]
        short_entry = short_breakout and bearish_bias and strong_bearish and volume_spike[i]
        
        # Exit logic: price returns to weekly pivot or opposite breakout
        long_exit = close[i] <= pivot_aligned[i] or short_breakout
        short_exit = close[i] >= pivot_aligned[i] or long_breakout
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_donchian_breakout_weekly_pivot_volume_v1"
timeframe = "1d"
leverage = 1.0