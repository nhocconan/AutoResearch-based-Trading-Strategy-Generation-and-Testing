#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Power with weekly trend filter and volume confirmation
# Uses Elder Ray (Bull Power = High - EMA(13), Bear Power = Low - EMA(13)) from 1d
# Takes longs when Bull Power > 0 and weekly EMA(50) rising, shorts when Bear Power < 0 and weekly EMA(50) falling
# Requires volume > 1.5x 20-period average for entry
# Exits when power signals reverse or volume drops
# Designed to work in both bull (strong Bull Power) and bear (strong Bear Power) markets
# Target: 12-30 trades per year with position size 0.25

name = "6h_ElderRay_WeeklyTrend_Volume"
timeframe = "6h"
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
    
    # Get daily data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate EMA(13) for Elder Ray
    close_1d = df_1d['close']
    ema_13_1d = close_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = df_1d['high'].values - ema_13_1d  # High - EMA(13)
    bear_power = df_1d['low'].values - ema_13_1d   # Low - EMA(13)
    
    # Align daily Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    close_1w = df_1w['close']
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_prev = np.roll(ema_50_1w, 1)
    ema_50_1w_prev[0] = ema_50_1w[0]
    ema_rising = ema_50_1w > ema_50_1w_prev
    ema_falling = ema_50_1w < ema_50_1w_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_falling)
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bull Power > 0 + weekly EMA rising + volume spike
            if (bull_power_aligned[i] > 0 and 
                ema_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0 + weekly EMA falling + volume spike
            elif (bear_power_aligned[i] < 0 and 
                  ema_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power <= 0 OR weekly EMA not rising
            if (bull_power_aligned[i] <= 0) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power >= 0 OR weekly EMA not falling
            if (bear_power_aligned[i] >= 0) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals