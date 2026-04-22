#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with Donchian(20) breakout and volume confirmation
# Long when: price breaks above Donchian(20) upper band, CHOP > 61.8 (trending regime), volume spike
# Short when: price breaks below Donchian(20) lower band, CHOP > 61.8 (trending regime), volume spike
# Exit when: price returns to Donchian midpoint or CHOP < 38.2 (range regime)
# Choppiness Index filters out false breakouts in ranging markets, improving breakout quality in both bull and bear markets.
# Target: 20-30 trades/year per symbol to minimize fee drag while maintaining edge.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Choppiness Index (14-period)
    chop_period = 14
    atr = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=chop_period, min_periods=chop_period).mean().values
    sum_atr = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high_chop = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low_chop = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(sum_atr / (highest_high_chop - lowest_low_chop)) / np.log10(chop_period)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout up + trending regime (CHOP > 61.8) + volume spike
            if (close[i] > highest_high[i] and 
                chop[i] > 61.8 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + trending regime (CHOP > 61.8) + volume spike
            elif (close[i] < lowest_low[i] and 
                  chop[i] > 61.8 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: return to Donchian midpoint OR range regime (CHOP < 38.2)
            if (close[i] < donchian_mid[i] and position == 1) or \
               (close[i] > donchian_mid[i] and position == -1) or \
               chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_DonchianBreakout_ChopRegime_VolumeSpike"
timeframe = "4h"
leverage = 1.0