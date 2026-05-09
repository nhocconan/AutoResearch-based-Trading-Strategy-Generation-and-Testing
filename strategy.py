#!/usr/bin/env python3
# Hypothesis: 4h Donchian breakout with 12h ATR volatility filter and volume confirmation
# Long when: price breaks above Donchian(20) high, 12h ATR(14) rising, volume > 1.5x 20-period average
# Short when: price breaks below Donchian(20) low, 12h ATR(14) falling, volume > 1.5x 20-period average
# Exit when: price crosses the midpoint of Donchian(20) channel or ATR reverses direction
# Position size: 0.25 to limit drawdown. Target: 25-50 trades/year.
# Designed to capture breakouts in trending markets while avoiding false signals in low volatility.

name = "4h_Donchian20_12hATR_Volume"
timeframe = "4h"
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
    
    # Donchian(20) channel
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Get 12h data for ATR volatility filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # 12h ATR(14) for volatility filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr_14_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_prev = np.roll(atr_14_12h, 1)
    atr_prev[0] = atr_14_12h[0]
    atr_rising = atr_14_12h > atr_prev
    atr_falling = atr_14_12h < atr_prev
    
    atr_rising_aligned = align_htf_to_ltf(prices, df_12h, atr_rising)
    atr_falling_aligned = align_htf_to_ltf(prices, df_12h, atr_falling)
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(atr_rising_aligned[i]) or np.isnan(atr_falling_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high + ATR rising + volume spike
            if (close[i] > donchian_high[i] and 
                atr_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + ATR falling + volume spike
            elif (close[i] < donchian_low[i] and 
                  atr_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian mid OR ATR turns falling
            if (close[i] < donchian_mid[i]) or (not atr_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian mid OR ATR turns rising
            if (close[i] > donchian_mid[i]) or (not atr_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals