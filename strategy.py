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
    
    # Get daily data for weekly pivot calculation and volume profile
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot levels from daily data (requires full week)
    # Use daily high/low/close of previous week for pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using last complete week)
    # For each day, we use the previous week's data
    pivot_points = np.full(len(close_1d), np.nan)
    r1 = np.full(len(close_1d), np.nan)
    s1 = np.full(len(close_1d), np.nan)
    r2 = np.full(len(close_1d), np.nan)
    s2 = np.full(len(close_1d), np.nan)
    
    for i in range(7, len(close_1d)):  # Start from 7th day to have full previous week
        # Previous week's data (7 days back to 1 day back)
        week_high = np.max(high_1d[i-7:i])
        week_low = np.min(low_1d[i-7:i])
        week_close = close_1d[i-1]  # Previous day's close as weekly close approximation
        
        pivot = (week_high + week_low + week_close) / 3.0
        pivot_points[i] = pivot
        r1[i] = 2 * pivot - week_low
        s1[i] = 2 * pivot - week_high
        r2[i] = pivot + (week_high - week_low)
        s2[i] = pivot - (week_high - week_low)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_points_aligned = align_htf_to_ltf(prices, df_1d, pivot_points)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Calculate 6-period RSI for overbought/oversold signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[5] = np.mean(gain[1:6])  # First average
    avg_loss[5] = np.mean(loss[1:6])
    
    for i in range(6, n):
        avg_gain[i] = (avg_gain[i-1] * 5 + gain[i]) / 6
        avg_loss[i] = (avg_loss[i-1] * 5 + loss[i]) / 6
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ma20_values = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ma20[:] = vol_ma20_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(pivot_points_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        vol_filter = volume[i] > vol_ma20[i] * 1.5
        
        # Fade at R3/S3 levels (using R1/S1 as primary levels, R2/S2 as extremes)
        # Long when price touches S1 with RSI < 30 (oversold) and volume confirmation
        long_setup = (close[i] <= s1_aligned[i] * 1.001) and (rsi[i] < 30) and vol_filter
        # Short when price touches R1 with RSI > 70 (overbought) and volume confirmation
        short_setup = (close[i] >= r1_aligned[i] * 0.999) and (rsi[i] > 70) and vol_filter
        
        # Breakout continuation at R4/S4 levels (using R2/S2)
        # Long breakout when price breaks above R2 with RSI > 50
        long_breakout = (close[i] > r2_aligned[i] * 1.001) and (rsi[i] > 50) and vol_filter
        # Short breakdown when price breaks below S2 with RSI < 50
        short_breakout = (close[i] < s2_aligned[i] * 0.999) and (rsi[i] < 50) and vol_filter
        
        long_entry = long_setup or long_breakout
        short_entry = short_setup or short_breakout
        
        # Exit conditions: opposite touch or RSI extreme reversal
        long_exit = (close[i] >= r1_aligned[i] * 0.999) or (rsi[i] > 70)
        short_exit = (close[i] <= s1_aligned[i] * 1.001) or (rsi[i] < 30)
        
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

name = "6h_1d_weekly_pivot_rsi_fade_breakout_v1"
timeframe = "6h"
leverage = 1.0