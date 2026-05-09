#!/usr/bin/env python3
# Hypothesis: 12-hour Donchian breakout with 1-day trend filter and volume confirmation
# Uses Donchian(20) breakout on 12h timeframe, filtered by 1d EMA(34) trend and volume spike
# Volume spike defined as current volume > 1.5x 20-period average volume
# Only takes longs when price > 12h Donchian upper AND 1d EMA(34) rising AND volume spike
# Only takes shorts when price < 12h Donchian lower AND 1d EMA(34) falling AND volume spike
# Exits when price crosses the 12h Donchian midpoint or trend reverses
# Target: 12-37 trades per year with position size 0.25

name = "12h_Donchian_1dTrend_VolumeSpike"
timeframe = "12h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close']
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_prev = np.roll(ema_34_1d, 1)
    ema_34_1d_prev[0] = ema_34_1d[0]
    ema_rising = ema_34_1d > ema_34_1d_prev
    ema_falling = ema_34_1d < ema_34_1d_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling)
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high']
    low_12h = df_12h['low']
    donchian_upper = high_12h.rolling(window=20, min_periods=20).max()
    donchian_lower = low_12h.rolling(window=20, min_periods=20).min()
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    donchian_upper_values = donchian_upper.values
    donchian_lower_values = donchian_lower.values
    donchian_mid_values = donchian_mid.values
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_values)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_values)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid_values)
    
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
            # Enter long: price > 12h Donchian upper + 1d EMA rising + volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                ema_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < 12h Donchian lower + 1d EMA falling + volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  ema_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 12h Donchian midpoint OR trend turns down
            if (close[i] < donchian_mid_aligned[i]) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 12h Donchian midpoint OR trend turns up
            if (close[i] > donchian_mid_aligned[i]) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals