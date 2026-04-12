#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_donchian_breakout_v2"
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
    
    # Get daily data for 1D pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous day's data for 1D pivot (to avoid look-ahead)
    high_1d_prev = df_1d['high'].shift(1).values
    low_1d_prev = df_1d['low'].shift(1).values
    close_1d_prev = df_1d['close'].shift(1).values
    
    # Previous week's data for trend filter
    high_1w_prev = df_1w['high'].shift(1).values
    low_1w_prev = df_1w['low'].shift(1).values
    close_1w_prev = df_1w['close'].shift(1).values
    
    # Calculate 1D pivot and levels
    pivot_1d_prev = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
    range_1d_prev = high_1d_prev - low_1d_prev
    
    # Camarilla levels from previous day
    h3_1d = pivot_1d_prev + (range_1d_prev * 1.1 / 4)
    l3_1d = pivot_1d_prev - (range_1d_prev * 1.1 / 4)
    h4_1d = pivot_1d_prev + (range_1d_prev * 1.1 / 2)
    l4_1d = pivot_1d_prev - (range_1d_prev * 1.1 / 2)
    
    # Align 1D levels to 6H timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d_prev)
    
    # Weekly trend: price above/below weekly EMA21
    weekly_close_series = pd.Series(df_1w['close'].values)
    weekly_ema21 = weekly_close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_ema21_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema21)
    
    # Volume filter: 6H volume > 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(weekly_ema21_aligned[i]) or np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > weekly_ema21_aligned[i]
        weekly_downtrend = close[i] < weekly_ema21_aligned[i]
        
        # Long: price breaks above H4 with volume and weekly uptrend
        long_signal = (close[i] > h4_1d_aligned[i] and 
                      volume_ok[i] and 
                      weekly_uptrend)
        
        # Short: price breaks below L4 with volume and weekly downtrend
        short_signal = (close[i] < l4_1d_aligned[i] and 
                       volume_ok[i] and 
                       weekly_downtrend)
        
        # Exit: price returns to 1D pivot
        exit_long = close[i] < pivot_1d_aligned[i]
        exit_short = close[i] > pivot_1d_aligned[i]
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals