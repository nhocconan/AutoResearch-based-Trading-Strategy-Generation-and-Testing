#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike.
Long when price breaks above upper Donchian channel (20-period) AND price > 1d EMA50 (uptrend) AND volume > 1.8x average.
Short when price breaks below lower Donchian channel (20-period) AND price < 1d EMA50 (downtrend) AND volume > 1.8x average.
Exit when price reverts to middle Donchian channel (20-period midpoint) OR trend reverses (price crosses 1d EMA50).
Uses 12h timeframe to target ~12-37 trades/year, minimizing fee drag while capturing strong breakouts.
Works in both bull and bear markets by requiring trend confirmation via 1d EMA50 for breakout entries.
"""

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
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 for 1d trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian channels (20-period) on 12h timeframe
    # Upper = max(high, 20), Lower = min(low, 20), Middle = (Upper + Lower) / 2
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Volume average (20-period) on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1d_aligned[i]
        upper_val = donchian_upper[i]
        lower_val = donchian_lower[i]
        middle_val = donchian_middle[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND price > 1d EMA50 (uptrend) AND volume spike
            if (price > upper_val and price > ema50_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND price < 1d EMA50 (downtrend) AND volume spike
            elif (price < lower_val and price < ema50_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to middle Donchian OR price breaks below 1d EMA50 (trend reversal)
                if price <= middle_val or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to middle Donchian OR price breaks above 1d EMA50 (trend reversal)
                if price >= middle_val or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1dEMA50_Volume_Breakout"
timeframe = "12h"
leverage = 1.0