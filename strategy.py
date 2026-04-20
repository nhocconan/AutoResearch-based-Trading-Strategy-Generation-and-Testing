#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout + Volume Spike + Choppiness Regime
# - Long: Price breaks above Donchian(20) high + volume > 1.5x average + chop > 61.8 (range)
# - Short: Price breaks below Donchian(20) low + volume > 1.5x average + chop > 61.8
# - Exit: Opposite Donchian break or volume drops below average
# - Choppiness filter ensures we only trade in ranging markets where breakouts are meaningful
# - Volume spike confirms breakout strength
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate indicators on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Average volume (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-period)
    atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((atr * 14) / (highest_high - lowest_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in indicators
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(avg_volume[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition (1.5x average)
        volume_spike = volume[i] > 1.5 * avg_volume[i]
        
        # Choppiness regime (range market)
        chop_range = chop[i] > 61.8
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_20[i-1]  # Using previous bar's high
        breakout_down = close[i] < low_20[i-1]  # Using previous bar's low
        
        if position == 0:
            # Long entry: Donchian breakout up + volume spike + chop range
            if breakout_up and volume_spike and chop_range:
                signals[i] = 0.25
                position = 1
            # Short entry: Donchian breakout down + volume spike + chop range
            elif breakout_down and volume_spike and chop_range:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Donchian breakout down or no volume spike
            if breakout_down or not volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Donchian breakout up or no volume spike
            if breakout_up or not volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_Chop"
timeframe = "4h"
leverage = 1.0