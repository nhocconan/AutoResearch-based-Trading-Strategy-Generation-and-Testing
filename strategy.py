#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above 6h Donchian upper (20) AND price > 1d EMA50 (uptrend) AND volume > 1.5x average.
Short when price breaks below 6h Donchian lower (20) AND price < 1d EMA50 (downtrend) AND volume > 1.5x average.
Exit when price reverts to 6h Donchian middle (median of upper/lower) OR trend reverses (price crosses 1d EMA50).
Uses 6h timeframe to target ~12-37 trades/year, avoiding fee drag while capturing strong breakouts.
Works in both bull and bear markets by requiring trend confirmation via 1d EMA50 for breakout entries.
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
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 for 1d trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 6h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max + low_min) / 2.0
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(donchian_mid[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1d_aligned[i]
        upper_val = high_max[i]
        lower_val = low_min[i]
        mid_val = donchian_mid[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above upper AND price > 1d EMA50 (uptrend) AND volume > 1.5x average
            if (price > upper_val and price > ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower AND price < 1d EMA50 (downtrend) AND volume > 1.5x average
            elif (price < lower_val and price < ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to mid OR price breaks below 1d EMA50 (trend reversal)
                if price <= mid_val or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to mid OR price breaks above 1d EMA50 (trend reversal)
                if price >= mid_val or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_1dEMA50_Volume_Breakout"
timeframe = "6h"
leverage = 1.0