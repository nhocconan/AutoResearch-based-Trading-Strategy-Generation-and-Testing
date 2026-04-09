#!/usr/bin/env python3
# 4h_donchian_20_volume_chop_regime_v1
# Hypothesis: 4h Donchian(20) breakout with volume confirmation (>2.0x 20-bar avg) and choppiness regime filter (CHOP > 61.8 = range, < 38.2 = trend).
# In ranging markets (CHOP > 61.8): mean reversion at Donchian bands (fade extremes).
# In trending markets (CHOP < 38.2): breakout continuation (buy high, sell low).
# Uses discrete position sizing (0.0, ±0.30) to minimize fee churn. ATR-based stoploss via signal=0 when price crosses opposite band.
# Designed to work in both bull and bear by adapting to regime. Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_volume_chop_regime_v1"
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
    
    # Choppiness Index (14-period) for regime detection
    def true_range(h, l, c):
        return np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    
    tr = true_range(high, low, close)
    tr[0] = np.nan  # First value undefined
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop_raw = 100 * np.log10(atr14.sum() / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop = pd.Series(chop_raw).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(chop[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr14[i]) or np.isnan(highest_high_14[i]) or np.isnan(lowest_low_14[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR regime shifts to ranging (mean reversion exit)
            if close[i] < lowest_low[i] or (is_ranging and close[i] < (highest_high[i] + lowest_low[i]) / 2):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR regime shifts to ranging (mean reversion exit)
            if close[i] > highest_high[i] or (is_ranging and close[i] > (highest_high[i] + lowest_low[i]) / 2):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed:
                if is_trending:
                    # Trending market: breakout continuation
                    if close[i] > highest_high[i]:
                        position = 1
                        signals[i] = 0.30
                    elif close[i] < lowest_low[i]:
                        position = -1
                        signals[i] = -0.30
                elif is_ranging:
                    # Ranging market: mean reversion at extremes
                    if close[i] < lowest_low[i]:
                        position = 1
                        signals[i] = 0.30
                    elif close[i] > highest_high[i]:
                        position = -1
                        signals[i] = -0.30
    
    return signals