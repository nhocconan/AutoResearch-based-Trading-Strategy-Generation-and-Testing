#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + ATR regime filter
# Donchian breakouts capture strong momentum moves. Volume confirmation ensures participation.
# ATR-based choppiness filter avoids whipsaws in ranging markets (CHOP > 61.8 = range, < 38.2 = trend).
# Works in bull markets (breakouts with volume) and bear markets (breakdowns with volume).
# Discrete sizing 0.25 targets ~100-180 total trades over 4 years (25-45/year) for 4h timeframe.

name = "4h_Donchian20_Volume_ATRRegime"
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
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ATR(14) for regime filter and stoploss
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # True Range for choppiness calculation (using ATR directly)
    # Choppiness Index: CHOP = 100 * log10(sum(ATR(14)) / (log10(lookback) * (HH(lookback) - LL(lookback))))
    # Simplified regime: use ATR ratio to detect choppy vs trending
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_ma_50  # High ratio = volatile/trending, Low ratio = choppy
    # Invert for regime: CHOP-like (>1 = trending, <1 = choppy)
    # We'll use: trending when ATR_ratio > 1.2, choppy when ATR_ratio < 0.8
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_ratio[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band + volume spike + trending regime
            if close[i] > highest_high[i] and volume[i] > (1.5 * vol_ema_20[i]) and atr_ratio[i] > 1.2:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower band + volume spike + trending regime
            elif close[i] < lowest_low[i] and volume[i] > (1.5 * vol_ema_20[i]) and atr_ratio[i] > 1.2:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian middle OR ATR regime turns choppy
            middle = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < middle or atr_ratio[i] < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian middle OR ATR regime turns choppy
            middle = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > middle or atr_ratio[i] < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals