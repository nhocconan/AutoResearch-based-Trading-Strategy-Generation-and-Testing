#!/usr/bin/env python3
# 4h_donchian_20_volume_chop_regime_v3
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and choppiness regime filter.
# In trending markets (CHOP < 38.2): breakout above/below Donchian(20) with volume > 1.5x 20-period average.
# In ranging markets (CHOP > 61.8): mean reversion at Donchian(20) bands with volume confirmation.
# Uses discrete position sizing (±0.25) to minimize fee churn. Target: 75-200 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_volume_chop_regime_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (highest_high_20 + lowest_low_20) / 2
    
    # Choppiness Index (14-period)
    def choppiness_index(high, low, close, window=14):
        atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=window, min_periods=window).sum()
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max()
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min()
        chop = 100 * np.log10(atr / (highest_high - lowest_low)) / np.log10(window)
        return chop.fillna(50).values
    
    chop = choppiness_index(high, low, close, 14)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(donchian_middle[i]) or np.isnan(chop[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian middle OR stoploss via signal=0
            if close[i] < donchian_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian middle OR stoploss via signal=0
            if close[i] > donchian_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Trending regime: CHOP < 38.2 -> breakout
                if chop[i] < 38.2:
                    if close[i] > highest_high_20[i]:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] < lowest_low_20[i]:
                        position = -1
                        signals[i] = -0.25
                # Ranging regime: CHOP > 61.8 -> mean reversion at bands
                elif chop[i] > 61.8:
                    if close[i] < lowest_low_20[i] and close[i] < donchian_middle[i]:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] > highest_high_20[i] and close[i] > donchian_middle[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals