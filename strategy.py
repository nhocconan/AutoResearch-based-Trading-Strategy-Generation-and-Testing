#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with weekly trend filter and volume confirmation.
# Long when price breaks above 20-day high AND weekly EMA34 rising AND volume > 2x 20-day average.
# Short when price breaks below 20-day low AND weekly EMA34 falling AND volume > 2x 20-day average.
# Exit when price crosses back inside the 20-day range.
# This strategy captures momentum breakouts aligned with weekly trend and institutional volume.
# Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag.
# Works in both bull and bear markets by following the weekly trend direction.

name = "1d_DonchianBreakout_WeeklyEMA34_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 20-day Donchian channels (using daily high/low)
    high_max20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Weekly EMA34 direction
    ema34_rising = np.zeros_like(ema34_1w_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1w_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1w_aligned[1:] > ema34_1w_aligned[:-1]
    ema34_falling[1:] = ema34_1w_aligned[1:] < ema34_1w_aligned[:-1]
    
    # Volume filter: current volume > 2x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Sufficient warmup for weekly EMA34 and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(high_max20[i]) or np.isnan(low_min20[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(ema34_rising[i]) or 
            np.isnan(ema34_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above 20-day high, weekly EMA34 rising, volume filter
            long_cond = (close[i] > high_max20[i]) and ema34_rising[i] and volume_filter[i]
            # Short conditions: price breaks below 20-day low, weekly EMA34 falling, volume filter
            short_cond = (close[i] < low_min20[i]) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below 20-day low
            if close[i] < low_min20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above 20-day high
            if close[i] > high_max20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals