#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 4h volume spike (volume > 2.0x 20-bar MA) and 1d EMA50 trend filter.
Long when price breaks above Donchian upper with volume confirmation AND 1d EMA50 rising.
Short when price breaks below Donchian lower with volume confirmation AND 1d EMA50 falling.
Exit when price touches the opposite Donchian level or when 1d EMA50 flips direction.
Uses 1d for EMA50 trend filter and 4h for Donchian/volume.
Designed to capture strong breakouts with volume confirmation in trending markets while filtering counter-trend noise.
Target: 20-35 trades/year per symbol (80-140 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_rising = ema_50_1d > np.roll(ema_50_1d, 1)
    ema_50_falling = ema_50_1d < np.roll(ema_50_1d, 1)
    ema_50_rising[0] = False
    ema_50_falling[0] = False
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_50_falling)
    
    # 4h Donchian(20) and volume MA
    period = 20
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for 1d EMA50 and 4h indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_rising_aligned[i]) or 
            np.isnan(ema_50_falling_aligned[i]) or
            np.isnan(highest[i]) or
            np.isnan(lowest[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-bar MA
        volume_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        # Breakout conditions
        breakout_upper = close[i] > highest[i]
        breakout_lower = close[i] < lowest[i]
        
        if position == 0:
            # Long: break above upper with volume confirmation and rising 1d EMA
            if (breakout_upper and volume_confirmed and ema_50_rising_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower with volume confirmation and falling 1d EMA
            elif (breakout_lower and volume_confirmed and ema_50_falling_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches lower Donchian OR 1d EMA50 falls
            if (close[i] <= lowest[i]) or (not ema_50_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches upper Donchian OR 1d EMA50 rises
            if (close[i] >= highest[i]) or (not ema_50_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_1dEMA50_Trend"
timeframe = "4h"
leverage = 1.0