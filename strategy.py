#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index (Chop) regime filter + 4h Donchian breakout + volume spike
# Chop > 61.8 = ranging (mean revert at Donchian bands), Chop < 38.2 = trending (follow Donchian breakout)
# Long: Chop < 38.2 (trending) + close > upper Donchian(20) + volume spike
# Short: Chop < 38.2 (trending) + close < lower Donchian(20) + volume spike
# Exit: Chop > 61.8 (range) or opposite Donchian breakout
# Volume: 4h volume > 1.5x 20-bar average to filter weak breakouts
# Chop regime prevents whipsaws in ranging markets, Donchian provides clear breakout levels, volume confirms strength

name = "4h_Chop_Donchian_Breakout_Volume"
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
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 4h Choppiness Index (14-period)
    chop_period = 14
    atr = pd.Series(np.maximum(high - low, np.maximum(high - np.roll(close, 1), low - np.roll(close, 1)))).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high_chop = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low_chop = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(atr / (highest_high_chop - lowest_low_chop)) / np.log10(chop_period)
    
    # Calculate 4h volume moving average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, chop_period, 20)  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 4h volume > 1.5x 20-bar average
        vol_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for breakout in trending regime (Chop < 38.2)
            if chop[i] < 38.2 and vol_filter:
                # Long: close above upper Donchian
                if close[i] > highest_high[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: close below lower Donchian
                elif close[i] < lowest_low[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: choppy regime (Chop > 61.8) or opposite breakout
            if chop[i] > 61.8 or close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: choppy regime (Chop > 61.8) or opposite breakout
            if chop[i] > 61.8 or close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals