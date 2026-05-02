#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + choppiness regime filter
# Long: price breaks above Donchian upper channel + volume spike + chop > 61.8 (range) for mean reversion
# Short: price breaks below Donchian lower channel + volume spike + chop > 61.8 (range) for mean reversion
# Uses discrete sizing 0.25 to balance profit and fee drag. Target: 20-50 trades/year.
# Works in both bull and bear markets by fading breakouts in ranging regimes (chop > 61.8)

name = "4h_Donchian20_Breakout_Volume_Chop"
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
    
    # Calculate Donchian channels (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Choppiness Index (14-period) - range detection
    def choppiness_index(high, low, close, window=14):
        atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1))))
        atr[0] = high[0] - low[0]  # first ATR
        atr_sum = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        highest_lowest_diff = highest_high - lowest_low
        
        chop = np.where(
            (highest_lowest_diff != 0) & (atr_sum != 0),
            100 * np.log10(atr_sum / highest_lowest_diff) / np.log10(window),
            50.0
        )
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    chop_range = chop > 61.8  # ranging market (mean reversion regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above upper Donchian + volume spike + ranging market
            if (close[i] > high_ma[i] and 
                volume_spike[i] and 
                chop_range[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian + volume spike + ranging market
            elif (close[i] < low_ma[i] and 
                  volume_spike[i] and 
                  chop_range[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below lower Donchian (mean reversion) OR chop exits ranging regime
            if close[i] < low_ma[i] or not chop_range[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian (mean reversion) OR chop exits ranging regime
            if close[i] > high_ma[i] or not chop_range[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals