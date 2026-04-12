#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot levels and ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily 14-period ATR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_1d = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        atr14_1d[i] = np.nanmean(tr_1d[i-13:i+1])
    
    # Calculate weekly pivot points from previous week (using daily data)
    # Group daily data into weeks and calculate pivot: (H+L+C)/3
    # We'll use rolling window of 5 days (approximate week) for pivot calculation
    if len(df_1d) >= 5:
        # Calculate weekly high, low, close using 5-day rolling window
        week_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
        week_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
        week_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
        
        # Pivot point = (week_high + week_low + week_close) / 3
        pivot_point = (week_high + week_low + week_close) / 3.0
        
        # Support and resistance levels
        s1 = (2 * pivot_point) - week_high
        r1 = (2 * pivot_point) - week_low
        s2 = pivot_point - (week_high - week_low)
        r2 = pivot_point + (week_high - week_low)
        
        # Align to 4h timeframe
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
        s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
        r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
        s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
        r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
        atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    else:
        return np.zeros(n)
    
    # Calculate 4h ATR for volatility filter
    tr1_4h = np.abs(high - low)
    tr2_4h = np.abs(high - np.roll(close, 1))
    tr3_4h = np.abs(low - np.roll(close, 1))
    tr1_4h[0] = tr2_4h[0] = tr3_4h[0] = np.nan
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr10_4h = np.full(n, np.nan)
    for i in range(9, n):
        atr10_4h[i] = np.nanmean(tr_4h[i-9:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(atr10_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current 4h ATR > 0.8x weekly ATR (avoid extremely low volatility)
        vol_filter = atr10_4h[i] > atr14_1d_aligned[i] * 0.8
        
        # Price position relative to pivot and support/resistance levels
        price_above_pivot = close[i] > pivot_aligned[i]
        price_below_pivot = close[i] < pivot_aligned[i]
        price_above_r1 = close[i] > r1_aligned[i]
        price_below_s1 = close[i] < s1_aligned[i]
        price_above_r2 = close[i] > r2_aligned[i]
        price_below_s2 = close[i] < s2_aligned[i]
        
        # Entry conditions: 
        # Long when price breaks above R1 with volatility expansion (bullish breakout)
        # Short when price breaks below S1 with volatility expansion (bearish breakdown)
        long_entry = price_above_r1 and vol_filter
        short_entry = price_below_s1 and vol_filter
        
        # Exit conditions:
        # Long exit: price returns below pivot OR volatility contracts significantly
        # Short exit: price returns above pivot OR volatility contracts significantly
        long_exit = (price_below_pivot) or (atr10_4h[i] < atr14_1d_aligned[i] * 0.5)
        short_exit = (price_above_pivot) or (atr10_4h[i] < atr14_1d_aligned[i] * 0.5)
        
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

name = "4h_1d_weekly_pivot_breakout_vol_filter_v1"
timeframe = "4h"
leverage = 1.0