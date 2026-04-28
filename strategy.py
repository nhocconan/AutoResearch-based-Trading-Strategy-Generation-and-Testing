# 12h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R3/S3 breakouts on 12h with 1d EMA50 trend filter and volume confirmation capture institutional moves.
# Works in bull/bear: breakouts capture momentum in trends, volume filter prevents fakeouts in ranges.
# Target: 15-30 trades/year on 12h timeframe to minimize fee drag.
# Uses 1d Camarilla levels and EMA50, aligned to 12h chart. No look-ahead bias.

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
    
    # Get daily data for Camarilla pivots and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R3, S3)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3_1d = pivot_1d + (range_1d * 1.1 / 2.0)
    s3_1d = pivot_1d - (range_1d * 1.1 / 2.0)
    
    # Calculate daily EMA(50) for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate average volume over 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or
            np.isnan(ema50_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Volume filter: current volume above average
        vol_filter = volume[i] > vol_ma[i]
        
        # Breakout conditions: price breaks R3/S3 with volume and trend
        long_breakout = close[i] > r3_aligned[i]
        short_breakout = close[i] < s3_aligned[i]
        
        long_entry = long_breakout and uptrend and vol_filter
        short_entry = short_breakout and downtrend and vol_filter
        
        # Exit conditions: price returns to pivot level or trend reverses
        pivot_1d = (high_1d + low_1d + close_1d) / 3.0
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
        long_exit = close[i] < pivot_aligned[i] or not uptrend
        short_exit = close[i] > pivot_aligned[i] or not downtrend
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0