#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla H3/L3 breakout with 1d weekly pivot bias and volume confirmation
    # Camarilla H3/L3 provides precise intraday support/resistance levels
    # Weekly pivot from 1d timeframe gives institutional bias for multi-day context
    # Volume > 1.8x 20-period average confirms institutional participation
    # Target: 15-30 trades/year (60-120 total over 4 years) for low fee drag
    
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
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Calculate 4h Camarilla levels (H3/L3) - using previous day's range
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Previous day's high, low, close for Camarilla calculation
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        camarilla_h3[i] = prev_close + 1.1 * (prev_high - prev_low) / 4
        camarilla_l3[i] = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Volume confirmation (>1.8x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.8 * vol_ma)
    
    # Align all 1d indicators to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions at Camarilla H3/L3 levels
        long_breakout = close[i] > camarilla_h3[i]
        short_breakout = close[i] < camarilla_l3[i]
        
        # Weekly pivot bias from 1d timeframe
        bullish_bias = close[i] > pivot_aligned[i]
        bearish_bias = close[i] < pivot_aligned[i]
        
        # Strong bullish/bearish conditions (beyond R1/S1)
        strong_bullish = close[i] > r1_aligned[i]
        strong_bearish = close[i] < s1_aligned[i]
        
        # Entry logic: Camarilla breakout + weekly pivot alignment + volume confirmation
        long_entry = long_breakout and bullish_bias and strong_bullish and volume_spike[i]
        short_entry = short_breakout and bearish_bias and strong_bearish and volume_spike[i]
        
        # Exit logic: price returns to weekly pivot or opposite Camarilla breakout
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

name = "4h_1d_camarilla_h3l3_weekly_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0