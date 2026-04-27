#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla pivot point reversal with volume confirmation and 12-hour trend filter.
Trades reversals at key Camarilla levels (L3, L4, H3, H4) in the direction of the 12-hour trend.
Designed to work in both bull and bear markets by using the 12-hour trend as filter.
Target: 20-50 trades/year per symbol (80-200 total over 4 years) to minimize fee drag.
"""
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
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla levels: H4, H3, L3, L4
    pivot = (high_prev + low_prev + close_prev) / 3
    range_val = high_prev - low_prev
    
    H4 = close_prev + range_val * 1.1 / 2
    H3 = close_prev + range_val * 1.1 / 4
    L3 = close_prev - range_val * 1.1 / 4
    L4 = close_prev - range_val * 1.1 / 2
    
    # Align Camarilla levels
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Get 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12-hour EMA(50) for trend
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 4-hour data for volume filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4-hour volume MA(20)
    vol_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Camarilla levels, volume MA, and 12h EMA
    start_idx = max(2, 20, 50)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H4_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(vol_ma_20_4h_aligned[i]) or np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        H4 = H4_aligned[i]
        H3 = H3_aligned[i]
        L3 = L3_aligned[i]
        L4 = L4_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_4h_aligned[i]
        trend_12h = ema_50_12h_aligned[i]
        
        # Volume filter: volume > 1.5x 4h average (moderate to balance trades)
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: Camarilla level reversal with volume and 12h trend alignment
        if position == 0:
            # Long: bounce from L3 or L4 + volume + 12h uptrend
            if ((close[i] > L3 and close[i] <= L3 * 1.001) or 
                (close[i] > L4 and close[i] <= L4 * 1.001)) and vol_filter and close[i] > trend_12h:
                signals[i] = size
                position = 1
            # Short: rejection from H3 or H4 + volume + 12h downtrend
            elif ((close[i] < H3 and close[i] >= H3 * 0.999) or 
                  (close[i] < H4 and close[i] >= H4 * 0.999)) and vol_filter and close[i] < trend_12h:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below 12h EMA or at opposite Camarilla level (H3)
            if close[i] < trend_12h or close[i] >= H3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above 12h EMA or at opposite Camarilla level (L3)
            if close[i] > trend_12h or close[i] <= L3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_CamarillaReversal_Volume_12hTrendFilter"
timeframe = "4h"
leverage = 1.0