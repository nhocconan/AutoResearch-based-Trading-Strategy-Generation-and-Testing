#!/usr/bin/env python3
# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation
# Uses Donchian(20) breakout on 4h timeframe, filtered by 12h EMA trend and volume spike
# Volume spike defined as current volume > 1.5x 20-period average volume
# Only takes longs when price > 4h Donchian upper AND 12h EMA(50) rising AND volume spike
# Only takes shorts when price < 4h Donchian lower AND 12h EMA(50) falling AND volume spike
# Exits when price crosses the 4h Donchian midpoint or trend reverses
# Target: 20-50 trades per year with position size 0.25

name = "4h_Donchian_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close']
    ema_50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_prev = np.roll(ema_50_12h, 1)
    ema_50_12h_prev[0] = ema_50_12h[0]
    ema_rising = ema_50_12h > ema_50_12h_prev
    ema_falling = ema_50_12h < ema_50_12h_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_falling)
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high']
    low_4h = df_4h['low']
    donchian_upper = high_4h.rolling(window=20, min_periods=20).max()
    donchian_lower = low_4h.rolling(window=20, min_periods=20).min()
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    donchian_upper_values = donchian_upper.values
    donchian_lower_values = donchian_lower.values
    donchian_mid_values = donchian_mid.values
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_values)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_values)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_values)
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > 4h Donchian upper + 12h EMA rising + volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                ema_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < 4h Donchian lower + 12h EMA falling + volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  ema_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 4h Donchian midpoint OR trend turns down
            if (close[i] < donchian_mid_aligned[i]) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 4h Donchian midpoint OR trend turns up
            if (close[i] > donchian_mid_aligned[i]) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals