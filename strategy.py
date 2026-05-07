#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d Donchian(20) breakout with volume confirmation.
# Long when price breaks above Donchian(20) high AND Choppiness Index > 61.8 (range) AND volume spike.
# Short when price breaks below Donchian(20) low AND Choppiness Index > 61.8 (range) AND volume spike.
# Uses mean reversion in ranging markets (Chop > 61.8) with Donchian breakouts for entry and volume for confirmation.
# Designed for low trade frequency (target: 20-30/year) to minimize fee drag and improve generalization.
# Works in ranging markets via mean reversion at channel extremes and avoids trending markets (Chop < 38.2) to reduce whipsaw.
name = "4h_ChopRange_Donchian20_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Choppiness Index (14) - range: 0-100, >61.8 = ranging, <38.2 = trending
    atr = np.abs(high - low)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop[highest_high == lowest_low] = 50  # avoid division by zero
    
    # Chop regime: ranging (Chop > 61.8)
    chop_range = chop > 61.8
    
    # Donchian(20) channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Donchian breakout signals
    long_breakout = close > highest_high_20
    short_breakout = close < lowest_low_20
    
    # Load 1d data for Donchian reference (optional, using same 4h for simplicity)
    # Volume confirmation: current volume > 2.0 * 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(chop_range[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout up + Chop range + volume spike
            long_condition = long_breakout[i] and chop_range[i] and volume_spike[i]
            # Short: Donchian breakout down + Chop range + volume spike
            short_condition = short_breakout[i] and chop_range[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below Donchian(20) low or Chop becomes trending (< 38.2)
            if close[i] < lowest_low_20[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above Donchian(20) high or Chop becomes trending (< 38.2)
            if close[i] > highest_high_20[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals