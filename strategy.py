#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above upper Donchian channel AND price > 1w EMA50 (uptrend) AND volume > 1.5x average.
Short when price breaks below lower Donchian channel AND price < 1w EMA50 (downtrend) AND volume > 1.5x average.
Exit when price reverts to the middle of the Donchian channel or trend reverses (price crosses 1w EMA50).
Uses 12h timeframe to target ~12-37 trades/year, avoiding fee drag while capturing strong breakouts.
Works in both bull and bear markets by requiring trend confirmation via 1w EMA50 for breakout entries.
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
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 for 1w trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian(20) channels on 12h timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (highest_high + lowest_low) / 2.0
    
    # Volume average (20-period) on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(donchian_middle[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1w_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        middle_channel = donchian_middle[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND price > 1w EMA50 (uptrend) AND volume confirmation
            if (price > upper_channel and price > ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND price < 1w EMA50 (downtrend) AND volume confirmation
            elif (price < lower_channel and price < ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to middle channel OR price breaks below 1w EMA50 (trend reversal)
                if price <= middle_channel or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to middle channel OR price breaks above 1w EMA50 (trend reversal)
                if price >= middle_channel or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1wEMA50_Volume_Breakout"
timeframe = "12h"
leverage = 1.0