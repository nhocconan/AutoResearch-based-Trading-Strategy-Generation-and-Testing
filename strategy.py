#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian(20) breakout with volume confirmation and choppiness regime filter.
# Long when price breaks above Donchian(20) high AND volume > 1.5x MA20 volume AND CHOP(14) < 38.2 (trending).
# Short when price breaks below Donchian(20) low AND volume > 1.5x MA20 volume AND CHOP(14) < 38.2 (trending).
# Exit when price crosses Donchian(20) midpoint OR CHOP(14) > 61.8 (choppy regime).
# Uses discrete position size 0.30. Donchian provides clear structure, volume confirms breakout strength,
# chop filter avoids whipsaws in ranging markets. Works in bull markets (capture uptrend breakouts) and
# bear markets (capture downtrend breakdowns). 4h timeframe targets 75-200 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Primary Indicators (4h) ===
    # Donchian Channel (20)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2.0
    
    # Volume MA (20)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14)
    atr_14 = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    highest_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10(highest_14 - lowest_14) * np.sqrt(14)
    chop_num = np.log10(atr_14.sum())
    chop = 100 * chop_num / chop_denom
    # Handle division by zero or invalid values
    chop = np.where((highest_14 - lowest_14) > 0, chop, 50.0)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 20
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol = volume[i]
        highest = highest_20[i]
        lowest = lowest_20[i]
        midpoint = donchian_mid[i]
        vol_ma = vol_ma_20[i]
        chop_value = chop[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < Donchian midpoint OR chop > 61.8 (choppy)
            if (price < midpoint) or (chop_value > 61.8):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > Donchian midpoint OR chop > 61.8 (choppy)
            if (price > midpoint) or (chop_value > 61.8):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price > Donchian high AND volume > 1.5x MA20 volume AND chop < 38.2 (trending)
            if (price > highest) and (vol > 1.5 * vol_ma) and (chop_value < 38.2):
                signals[i] = 0.30
                position = 1
            
            # SHORT: price < Donchian low AND volume > 1.5x MA20 volume AND chop < 38.2 (trending)
            elif (price < lowest) and (vol > 1.5 * vol_ma) and (chop_value < 38.2):
                signals[i] = -0.30
                position = -1
        
        else:
            signals[i] = position * 0.30  # maintain position
    
    return signals

name = "4h_Donchian20_Volume_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0