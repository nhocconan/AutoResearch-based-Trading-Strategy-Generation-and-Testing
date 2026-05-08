#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volatility filter and volume confirmation.
# Long when price breaks above 20-period Donchian high AND 1d ATR ratio > 1.2 (low volatility regime) AND volume > 1.3x 20-period average.
# Short when price breaks below 20-period Donchian low AND 1d ATR ratio > 1.2 AND volume > 1.3x 20-period average.
# Exit when price crosses back inside the Donchian channel.
# Donchian provides clear breakout levels. ATR filter avoids choppy markets. Volume confirms breakout strength.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Donchian_20_1dATR_Ratio_Volume"
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
    
    # 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ATR(14) on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr_14 = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < 14:
            atr_14[i] = np.mean(tr[:i+1]) if i > 0 else tr[0]
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # ATR ratio: current ATR / 20-period ATR average (to detect low volatility)
    atr_ma20 = np.zeros_like(atr_14)
    for i in range(len(atr_14)):
        if i < 20:
            atr_ma20[i] = np.mean(atr_14[:i+1]) if i > 0 else atr_14[0]
        else:
            atr_ma20[i] = np.mean(atr_14[i-19:i+1])
    
    atr_ratio = atr_14 / atr_ma20
    atr_ratio[atr_ma20 == 0] = 1.0  # Avoid division by zero
    
    # Align ATR ratio to 12h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate Donchian channels (20-period) on 12h data
    high_max20 = np.zeros_like(high)
    low_min20 = np.zeros_like(low)
    
    for i in range(len(high)):
        start_idx = max(0, i - 19)
        high_max20[i] = np.max(high[start_idx:i+1])
        low_min20[i] = np.min(low[start_idx:i+1])
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma20 = np.zeros_like(volume)
    for i in range(len(volume)):
        start_idx = max(0, i - 19)
        vol_ma20[i] = np.mean(volume[start_idx:i+1])
    
    volume_filter = volume > (1.3 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Sufficient warmup for Donchian and ATR
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_max20[i]) or np.isnan(low_min20[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, low volatility (ATR ratio > 1.2), volume filter
            long_cond = (close[i] > high_max20[i]) and (atr_ratio_aligned[i] > 1.2) and volume_filter[i]
            # Short conditions: price breaks below Donchian low, low volatility, volume filter
            short_cond = (close[i] < low_min20[i]) and (atr_ratio_aligned[i] > 1.2) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian low
            if close[i] < low_min20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian high
            if close[i] > high_max20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals