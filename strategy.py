#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above Donchian upper (20) AND price > 1d EMA34 (uptrend) AND volume > 1.5x average.
Short when price breaks below Donchian lower (20) AND price < 1d EMA34 (downtrend) AND volume > 1.5x average.
Exit when price reverts to Donchian middle or trend reverses (price crosses 1d EMA34).
Uses 4h timeframe to target 75-200 trades over 4 years, avoiding fee drag while capturing strong breakouts.
Works in both bull and bear markets by requiring trend confirmation via 1d EMA34 for breakout entries.
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
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 for 1d trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Donchian Channel (20) on 4h timeframe
    dc_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    dc_middle = (dc_upper + dc_lower) / 2
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or np.isnan(dc_middle[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_1d_aligned[i]
        dc_upper_val = dc_upper[i]
        dc_lower_val = dc_lower[i]
        dc_middle_val = dc_middle[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND price > 1d EMA34 (uptrend) AND volume spike
            if (price > dc_upper_val and price > ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND price < 1d EMA34 (downtrend) AND volume spike
            elif (price < dc_lower_val and price < ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to Donchian middle OR price breaks below 1d EMA34 (trend reversal)
                if price <= dc_middle_val or price < ema34_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to Donchian middle OR price breaks above 1d EMA34 (trend reversal)
                if price >= dc_middle_val or price > ema34_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0