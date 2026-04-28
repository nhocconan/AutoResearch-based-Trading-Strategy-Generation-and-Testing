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
    
    # Get weekly data for trend and pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR (14)
    tr1 = high_1w[1:] - low_1w[:-1]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly high/low for pivot-like structure
    weekly_high = pd.Series(high_1w).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1w).rolling(window=5, min_periods=5).min().values
    
    # Align weekly indicators to 6h
    atr_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_aligned[i]) or 
            np.isnan(weekly_high_aligned[i]) or
            np.isnan(weekly_low_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly average
        weekly_mid = (weekly_high_aligned[i] + weekly_low_aligned[i]) / 2
        uptrend = close[i] > weekly_mid
        downtrend = close[i] < weekly_mid
        
        # Volatility filter: only trade when volatility is sufficient
        vol_filter = atr_aligned[i] > 0.5 * np.nanmedian(atr_aligned[max(0, i-50):i+1])
        
        # Volume filter: current volume above daily average
        vol_filter = vol_filter and volume[i] > vol_ma_aligned[i]
        
        # Breakout conditions: price breaks weekly high/low with volume
        long_breakout = close[i] > weekly_high_aligned[i] and vol_filter
        short_breakout = close[i] < weekly_low_aligned[i] and vol_filter
        
        # Entry conditions: breakout + volume
        long_entry = long_breakout
        short_entry = short_breakout
        
        # Exit conditions: price returns to weekly midpoint
        long_exit = close[i] < weekly_mid
        short_exit = close[i] > weekly_mid
        
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

name = "6h_WeeklyBreakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0