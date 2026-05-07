#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper AND 12h EMA50 rising AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower AND 12h EMA50 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back inside Donchian channel (below middle for longs, above middle for shorts).
# This targets volatility expansion with trend alignment to capture momentum while avoiding choppy markets.
# The 12h EMA50 filter ensures trading with higher timeframe trend. Volume confirms institutional participation.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following 12h trend direction.

name = "4h_DonchianBreakout_12hEMA50_Volume"
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
    
    # Donchian Channel (20)
    dc_length = 20
    high_roll = pd.Series(high).rolling(window=dc_length, min_periods=dc_length).max().values
    low_roll = pd.Series(low).rolling(window=dc_length, min_periods=dc_length).min().values
    upper_channel = high_roll
    lower_channel = low_roll
    middle_channel = (upper_channel + lower_channel) * 0.5
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 12h EMA50 direction
    ema50_rising = np.zeros_like(ema50_12h_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_12h_aligned, dtype=bool)
    ema50_rising[1:] = ema50_12h_aligned[1:] > ema50_12h_aligned[:-1]
    ema50_falling[1:] = ema50_12h_aligned[1:] < ema50_12h_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(dc_length, 50)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(middle_channel[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(ema50_rising[i]) or np.isnan(ema50_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian, 12h EMA50 rising, volume filter
            long_cond = (close[i] > upper_channel[i]) and ema50_rising[i] and volume_filter[i]
            # Short conditions: price breaks below lower Donchian, 12h EMA50 falling, volume filter
            short_cond = (close[i] < lower_channel[i]) and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back inside Donchian channel (below middle)
            if close[i] < middle_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back inside Donchian channel (above middle)
            if close[i] > middle_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals