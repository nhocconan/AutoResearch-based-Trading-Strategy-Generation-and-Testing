#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with Donchian(20) breakout and volume confirmation.
# Uses Choppiness Index (14) to identify trending (CHOP < 38.2) vs ranging (CHOP > 61.8) markets.
# In trending regimes: breakout of Donchian channel with volume confirmation.
# In ranging regimes: mean reversion at Donchian channel boundaries with volume confirmation.
# Designed to work in both bull and bear markets by adapting to market regime.
# Target: 20-50 trades/year to minimize fee drag.

name = "4h_Chop_Donchian_Breakout_MeanRev_Volume"
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
    
    # Choppiness Index (14)
    def choppiness_index(high, low, close, window=14):
        atr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        atr[0] = high[0] - low[0]  # first ATR
        tr_sum = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        range_max_min = highest_high - lowest_low
        chop = 100 * np.log10(tr_sum / (range_max_min * window)) / np.log10(window)
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    
    # Donchian channels (20)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            if chop[i] < 38.2:  # Trending regime
                # Long: break above Donchian high + volume spike
                long_cond = (close[i] > highest_high[i]) and volume_spike[i]
                # Short: break below Donchian low + volume spike
                short_cond = (close[i] < lowest_low[i]) and volume_spike[i]
            else:  # Ranging regime (chop > 38.2, could be choppy or ranging)
                # Long: pullback to Donchian low + volume spike (mean reversion)
                long_cond = (close[i] <= lowest_low[i] * 1.001) and volume_spike[i]  # near low
                # Short: pullback to Donchian high + volume spike (mean reversion)
                short_cond = (close[i] >= highest_high[i] * 0.999) and volume_spike[i]  # near high
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: opposite Donchian touch or loss of momentum
            if chop[i] < 38.2:  # trending: exit on opposite break
                if close[i] < lowest_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # ranging: exit at opposite extreme
                if close[i] >= highest_high[i] * 0.999:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit: opposite Donchian touch or loss of momentum
            if chop[i] < 38.2:  # trending: exit on opposite break
                if close[i] > highest_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # ranging: exit at opposite extreme
                if close[i] <= lowest_low[i] * 1.001:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals