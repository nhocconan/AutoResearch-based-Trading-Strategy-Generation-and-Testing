#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian(20) breakout with 12-hour EMA(50) trend filter and volume confirmation.
Long when price breaks above Donchian upper channel, 12h EMA50 rising, volume > 1.5x average.
Short when price breaks below Donchian lower channel, 12h EMA50 falling, volume > 1.5x average.
Exit when price returns to Donchian middle (mean) or volume drops below average.
Designed for low trade frequency (~20-40/year) to capture strong trends while minimizing whipsaws.
Works in both bull and bear markets by requiring trend confirmation from higher timeframe.
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
    
    # Load 12-hour data for EMA50 - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12-hour EMA(50)
    close_12h = df_12h['close'].values
    ema50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_prev = np.roll(ema50, 1)
    ema50_prev[0] = ema50[0]
    ema50_rising = ema50 > ema50_prev
    
    # Calculate Donchian(20) channels on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0
    
    # Volume average (20-period) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema50[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_rising_val = ema50_rising[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Breakout above upper channel, EMA50 rising, volume confirmation
            if (close[i] > high_20[i] and ema50_rising_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below lower channel, EMA50 falling, volume confirmation
            elif (close[i] < low_20[i] and not ema50_rising_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to middle OR volume drops below average
                if close[i] <= mid_20[i] or vol_current < vol_ma_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to middle OR volume drops below average
                if close[i] >= mid_20[i] or vol_current < vol_ma_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_12hEMA50_Volume_Breakout"
timeframe = "4h"
leverage = 1.0