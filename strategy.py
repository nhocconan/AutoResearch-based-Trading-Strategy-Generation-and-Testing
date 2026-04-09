#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v7
# Hypothesis: 4h Donchian channel breakout with volume confirmation and choppiness regime filter.
# Long when price breaks above 20-period upper band + volume spike + chop > 61.8 (range).
# Short when price breaks below 20-period lower band + volume spike + chop > 61.8 (range).
# Exits when price reverts to 20-period middle band or chop < 38.2 (trend).
# Works in bull/bear: chop filter avoids whipsaws in trends, volume confirms breakout validity.
# Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v7"
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
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper = high_roll
    lower = low_roll
    middle = (upper + lower) / 2.0
    
    # Choppiness Index (14-period)
    atr = pd.Series(np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=20, min_periods=20).max().values
    ll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    chop = 100 * np.log10(atr_sum / np.log10(14) / (hh - ll)) / np.log10(14)
    chop = np.where((hh - ll) > 0, chop, 50.0)  # avoid division by zero
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(chop[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reverts to middle band OR chop < 38.2 (trending)
            if close[i] <= middle[i] or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reverts to middle band OR chop < 38.2 (trending)
            if close[i] >= middle[i] or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed and chop[i] > 61.8:  # range regime
                # Check for breakout
                if close[i] > upper[i]:
                    # Breakout above upper band → long
                    position = 1
                    signals[i] = 0.25
                elif close[i] < lower[i]:
                    # Breakout below lower band → short
                    position = -1
                    signals[i] = -0.25
    
    return signals