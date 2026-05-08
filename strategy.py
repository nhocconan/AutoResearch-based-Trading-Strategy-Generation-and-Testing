#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA(34) trend filter and volume confirmation
# Go long when price breaks above Donchian(20) upper band and EMA(34) is rising
# Go short when price breaks below Donchian(20) lower band and EMA(34) is falling
# Requires volume > 1.5x 20-period average for confirmation
# Designed for low trade frequency (target: 12-37 trades/year) to avoid fee drag
# Works in both bull and bear markets via trend filter

name = "12h_Donchian20_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_prev = np.roll(ema34_1d, 1)  # Previous day's EMA for trend direction
    ema34_1d_prev[0] = ema34_1d[0]  # Handle first value
    ema34_1d_rising = ema34_1d > ema34_1d_prev
    ema34_1d_falling = ema34_1d < ema34_1d_prev
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_rising)
    ema34_1d_falling_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_falling)
    
    # Donchian(20) channels on 12h data
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_1d_falling_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_rising_val = ema34_1d_aligned[i]
        ema34_1d_falling_val = ema34_1d_falling_aligned[i]
        donchian_upper_val = donchian_upper[i]
        donchian_lower_val = donchian_lower[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian upper + uptrend + volume spike
            if (close[i] > donchian_upper_val and 
                ema34_1d_rising_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian lower + downtrend + volume spike
            elif (close[i] < donchian_lower_val and 
                  ema34_1d_falling_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower OR trend changes
            if close[i] < donchian_lower_val or not ema34_1d_rising_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian upper OR trend changes
            if close[i] > donchian_upper_val or not ema34_1d_falling_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals