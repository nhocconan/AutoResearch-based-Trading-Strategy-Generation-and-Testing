#!/usr/bin/env python3
# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high, 1d EMA50 is rising, and volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) low, 1d EMA50 is falling, and volume > 1.5x 20-period average.
# Exit when price crosses back below/above Donchian midpoint.
# Position size: 0.25 (25% of capital) to balance return and drawdown.
# Designed to work in trending markets via 1d EMA filter and capture breakouts with volume confirmation.

name = "4h_Donchian_1dEMA_Volume_Filter"
timeframe = "4h"
leverage = 1.0

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
    
    # 4h Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (high_20 + low_20) / 2
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_prev = np.roll(ema50_1d, 1)
    ema50_1d_prev[0] = ema50_1d[0]  # handle first value
    ema50_rising = ema50_1d > ema50_1d_prev
    ema50_falling = ema50_1d < ema50_1d_prev
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema50_falling)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_rising_aligned[i]) or 
            np.isnan(ema50_falling_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, 1d EMA50 rising, volume spike
            if (close[i] > high_20[i] and 
                ema50_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, 1d EMA50 falling, volume spike
            elif (close[i] < low_20[i] and 
                  ema50_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint
            if close[i] < donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint
            if close[i] > donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals