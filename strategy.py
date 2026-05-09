#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA(50) trend filter and volume spike confirmation
# Long when: price breaks above Donchian high(20), 1d EMA(50) rising, volume > 2.0x 20-period average
# Short when: price breaks below Donchian low(20), 1d EMA(50) falling, volume > 2.0x 20-period average
# Exit when: price crosses Donchian midpoint OR trend reverses
# Position size: 0.28 to balance risk and return. Target: 15-30 trades/year.
# Designed to work in both bull (breakouts) and bear (mean-reversion at extremes) markets.

name = "12h_Donchian20_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_prev = np.roll(ema_50_1d, 1)
    ema_50_1d_prev[0] = ema_50_1d[0]
    ema_rising = ema_50_1d > ema_50_1d_prev
    ema_falling = ema_50_1d < ema_50_1d_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling)
    
    # Calculate 12h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = high_max
    donchian_low = low_min
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume spike: current volume > 2.0x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(donchian_mid[i]) or np.isnan(ema_rising_aligned[i]) or
            np.isnan(ema_falling_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > Donchian high + 1d EMA rising + volume spike
            if (close[i] > donchian_high[i] and 
                ema_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.28
                position = 1
            # Enter short: price < Donchian low + 1d EMA falling + volume spike
            elif (close[i] < donchian_low[i] and 
                  ema_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.28
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint OR trend turns down
            if (close[i] < donchian_mid[i]) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint OR trend turns up
            if (close[i] > donchian_mid[i]) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals