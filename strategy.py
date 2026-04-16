#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Load weekly data ONCE
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR on 12h
    tr_12h = np.maximum(high_12h - low_12h,
                        np.maximum(np.abs(high_12h - np.roll(close_12h, 1)),
                                   np.abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate ATR on weekly
    tr_1w = np.maximum(high_1w - low_1w,
                       np.maximum(np.abs(high_1w - np.roll(close_1w, 1)),
                                  np.abs(low_1w - np.roll(close_1w, 1))))
    tr_1w[0] = high_1w[0] - low_1w[0]
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Weekly Donchian Channel (20) for trend
    highest_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_upper_1w = highest_20_1w
    donchian_lower_1w = lowest_20_1w
    donchian_upper_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_1w)
    donchian_lower_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_1w)
    
    # 12h Donchian Channel (20) for breakout
    highest_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_upper_12h = highest_20_12h
    donchian_lower_12h = lowest_20_12h
    
    # 12h volume spike detection
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = volume_12h / vol_ma_20_12h
    
    signals = np.zeros(n)
    warmup = 100
    
    # Position tracking
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is invalid
        if (np.isnan(atr_12h_aligned[i]) or np.isnan(atr_1w_aligned[i]) or
            np.isnan(donchian_upper_12h[i]) or np.isnan(donchian_lower_12h[i]) or
            np.isnan(donchian_upper_1w_aligned[i]) or np.isnan(donchian_lower_1w_aligned[i]) or
            np.isnan(vol_ratio_12h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        atr_12h_val = atr_12h_aligned[i]
        vol_ratio_val = vol_ratio_12h[i]
        
        # Exit logic
        if position == 1:  # Long
            if price < donchian_lower_12h[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short
            if price > donchian_upper_12h[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry logic (only when flat)
        if position == 0:
            # Long: 12h price breaks above 12h Donchian upper AND weekly trend is up AND volume spike
            if (price > donchian_upper_12h[i] and 
                close_12h[i] > donchian_upper_1w_aligned[i] and 
                vol_ratio_val > 2.0):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: 12h price breaks below 12h Donchian lower AND weekly trend is down AND volume spike
            elif (price < donchian_lower_12h[i] and 
                  close_12h[i] < donchian_lower_1w_aligned[i] and 
                  vol_ratio_val > 2.0):
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0