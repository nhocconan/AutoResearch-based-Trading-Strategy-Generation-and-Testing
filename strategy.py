#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with Volume Spike and Choppiness Filter
# Uses 20-period Donchian channels from 4h data. Long when price breaks above upper band,
# short when breaks below lower band. Entry requires volume > 2x median volume (spike) and
# Choppiness Index > 61.8 (ranging market) to avoid false breakouts in strong trends.
# Exit when price returns to opposite Donchian band or Choppiness < 38.2 (trending).
# Designed for low trade frequency (<50/year) with high edge in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index (14-period) on 4h
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(sum_tr / (hh - ll)) / np.log10(14)
    # Handle division by zero when hh == ll
    chop = np.where((hh - ll) == 0, 50, chop)
    
    # Volume spike filter: volume > 2x median of last 20 periods
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(chop[i]) or np.isnan(vol_median[i])):
            continue
        
        # Long entry: break above upper Donchian band + volume spike + chop > 61.8 (ranging)
        if (close[i] > highest_high[i] and
            volume[i] > 2 * vol_median[i] and
            chop[i] > 61.8 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: break below lower Donchian band + volume spike + chop > 61.8 (ranging)
        elif (close[i] < lowest_low[i] and
              volume[i] > 2 * vol_median[i] and
              chop[i] > 61.8 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: price returns to opposite band or chop < 38.2 (trending)
        elif position == 1 and (close[i] < lowest_low[i] or chop[i] < 38.2):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > highest_high[i] or chop[i] < 38.2):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Breakout_Volume_Chop"
timeframe = "4h"
leverage = 1.0