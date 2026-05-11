#!/usr/bin/env python3
name = "1D_1W_Camarilla_Pivot_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

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
    
    # Get daily and weekly data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    pivot = (daily_high + daily_low + daily_close) / 3
    range_hl = daily_high - daily_low
    
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H4 = close + range * 1.1/2
    # L4 = close - range * 1.1/2
    h4 = pivot + (range_hl * 1.1 / 2)
    l4 = pivot - (range_hl * 1.1 / 2)
    
    # Weekly trend filter
    weekly_close = df_1w['close'].values
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_trend = weekly_close > weekly_ema20
    
    # Align weekly trend and Camarilla levels to daily
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(weekly_trend_aligned[i]) or 
            np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly uptrend + close breaks above H4 + volume confirmation
            if (weekly_trend_aligned[i] and 
                close[i] > h4_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + close breaks below L4 + volume confirmation
            elif (not weekly_trend_aligned[i] and 
                  close[i] < l4_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly trend changes or close drops below pivot
            if (not weekly_trend_aligned[i] or close[i] < pivot[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly trend changes or close rises above pivot
            if (weekly_trend_aligned[i] or close[i] > pivot[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals