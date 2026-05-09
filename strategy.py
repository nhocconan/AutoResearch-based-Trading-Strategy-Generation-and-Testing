#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with daily ATR filter and volume confirmation
# Long when: price breaks above 20-period high, daily ATR(14) > 1.5x 50-period average, volume spike
# Short when: price breaks below 20-period low, daily ATR(14) > 1.5x 50-period average, volume spike
# Exit when: price crosses back through the 20-period midpoint OR ATR volatility collapses
# Position size: 0.25 to manage drawdown. Target: 20-40 trades/year.
# Designed to capture breakouts in trending markets while avoiding low-volatility whipsaws.

name = "4h_Donchian20_ATRFilter_Volume"
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
    
    # Get daily data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr2[1]  # Avoid NaN at index 0
    tr3[0] = tr3[1]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 50-period average of ATR
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_expansion = atr_14 > (1.5 * atr_ma_50)
    
    # Align daily ATR filter to 4h timeframe
    atr_expansion_aligned = align_htf_to_ltf(prices, df_1d, atr_expansion)
    
    # Donchian(20) channels on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(atr_expansion_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high + ATR expansion + volume spike
            if (close[i] > high_20[i] and 
                atr_expansion_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + ATR expansion + volume spike
            elif (close[i] < low_20[i] and 
                  atr_expansion_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below midpoint OR ATR expansion ends
            if (close[i] < donchian_mid[i]) or (not atr_expansion_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above midpoint OR ATR expansion ends
            if (close[i] > donchian_mid[i]) or (not atr_expansion_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals