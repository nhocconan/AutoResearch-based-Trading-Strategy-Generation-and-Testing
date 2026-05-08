#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper (20-period high) AND 1d EMA50 rising AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower (20-period low) AND 1d EMA50 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back inside Donchian channel or trailing stop triggered.
# Donchian provides clear trend-following structure. EMA50 filters higher timeframe trend.
# Volume confirms institutional participation. Target: 50-150 total trades over 4 years (12-37/year).

name = "4h_Donchian_20_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d EMA50 direction
    ema50_rising = np.zeros_like(ema50_1d_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_1d_aligned, dtype=bool)
    ema50_rising[1:] = ema50_1d_aligned[1:] > ema50_1d_aligned[:-1]
    ema50_falling[1:] = ema50_1d_aligned[1:] < ema50_1d_aligned[:-1]
    
    # Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # Trailing stop parameters
    atr_period = 14
    atr_multiplier = 3.0
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = max(50, 20)  # Sufficient warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_rising[i]) or 
            np.isnan(ema50_falling[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper, 1d EMA50 rising, volume filter
            long_cond = (close[i] > high_20[i]) and ema50_rising[i] and volume_filter[i]
            # Short conditions: price breaks below Donchian lower, 1d EMA50 falling, volume filter
            short_cond = (close[i] < low_20[i]) and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            elif short_cond:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            # Long exit: price crosses back inside Donchian or trailing stop
            exit_cond = (close[i] < low_20[i]) or (high[i] - highest_high_since_entry > atr_multiplier * atr[i])
            if exit_cond:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            # Short exit: price crosses back inside Donchian or trailing stop
            exit_cond = (close[i] > high_20[i]) or (lowest_low_since_entry - low[i] > atr_multiplier * atr[i])
            if exit_cond:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals