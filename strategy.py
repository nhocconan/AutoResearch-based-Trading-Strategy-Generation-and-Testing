#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
    # Uses 1w Donchian channels for trend direction, 6h Donchian(20) for breakout entries
    # Volume spike (>2.0x 20-period average) confirms institutional participation
    # Designed for low trade frequency (target: 12-37/year) to minimize fee drag
    # Weekly pivot filter avoids counter-trend trades in ranging markets
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot and Donchian trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w weekly pivot points (standard floor trader pivots)
    pivot_1w = np.full(len(df_1w), np.nan)
    r1_1w = np.full(len(df_1w), np.nan)
    s1_1w = np.full(len(df_1w), np.nan)
    r2_1w = np.full(len(df_1w), np.nan)
    s2_1w = np.full(len(df_1w), np.nan)
    
    for i in range(1, len(df_1w)):
        high_val = high_1w[i-1]
        low_val = low_1w[i-1]
        close_val = close_1w[i-1]
        pivot_val = (high_val + low_val + close_val) / 3.0
        
        pivot_1w[i] = pivot_val
        r1_1w[i] = 2 * pivot_val - low_val
        s1_1w[i] = 2 * pivot_val - high_val
        r2_1w[i] = pivot_val + (high_val - low_val)
        s2_1w[i] = pivot_val - (high_val - low_val)
    
    # Calculate 1w Donchian(20) for trend filter (long-term trend)
    donchian_high_20_1w = np.full(len(df_1w), np.nan)
    donchian_low_20_1w = np.full(len(df_1w), np.nan)
    
    for i in range(20, len(df_1w)):
        donchian_high_20_1w[i] = np.max(high_1w[i-20:i])
        donchian_low_20_1w[i] = np.min(low_1w[i-20:i])
    
    # Get 6h data for entry Donchian breakout
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate 6h Donchian(20) for breakout entries
    donchian_high_20_6h = np.full(len(df_6h), np.nan)
    donchian_low_20_6h = np.full(len(df_6h), np.nan)
    
    for i in range(20, len(df_6h)):
        donchian_high_20_6h[i] = np.max(high_6h[i-20:i])
        donchian_low_20_6h[i] = np.min(low_6h[i-20:i])
    
    # Align HTF indicators to 6h timeframe (actually to LTF which is 6h here)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    donchian_high_20_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20_1w)
    donchian_low_20_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20_1w)
    donchian_high_20_6h_aligned = align_htf_to_ltf(prices, df_6h, donchian_high_20_6h)
    donchian_low_20_6h_aligned = align_htf_to_ltf(prices, df_6h, donchian_low_20_6h)
    
    # Volume confirmation: volume > 2.0 * 20-period average (6h)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or 
            np.isnan(s2_1w_aligned[i]) or np.isnan(donchian_high_20_1w_aligned[i]) or
            np.isnan(donchian_low_20_1w_aligned[i]) or np.isnan(donchian_high_20_6h_aligned[i]) or
            np.isnan(donchian_low_20_6h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend from 1w Donchian(20)
        weekly_uptrend = close[i] > donchian_high_20_1w_aligned[i]
        weekly_downtrend = close[i] < donchian_low_20_1w_aligned[i]
        
        # Entry logic: 6h Donchian breakout with weekly trend filter and volume
        long_entry = False
        short_entry = False
        
        # Long breakout: price breaks above 6h Donchian(20) in weekly uptrend with volume
        if weekly_uptrend:
            long_entry = (close[i] > donchian_high_20_6h_aligned[i]) and volume_spike[i]
        # Short breakout: price breaks below 6h Donchian(20) in weekly downtrend with volume
        elif weekly_downtrend:
            short_entry = (close[i] < donchian_low_20_6h_aligned[i]) and volume_spike[i]
        
        # Exit logic: opposite 6h Donchian level or weekly trend reversal
        long_exit = weekly_downtrend and close[i] < donchian_low_20_6h_aligned[i]
        short_exit = weekly_uptrend and close[i] > donchian_high_20_6h_aligned[i]
        
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

name = "6h_1w_donchian_breakout_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0