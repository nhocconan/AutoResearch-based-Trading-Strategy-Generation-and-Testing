#!/usr/bin/env python3
# 4h_donchian_volume_chop_regime_v2
# Hypothesis: 4h Donchian breakout with volume confirmation and choppiness regime filter.
# Donchian(20) breakout captures trend, volume > 1.5x average confirms momentum,
# Choppiness Index (14) < 38.2 ensures trending regime to avoid whipsaws.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 20-50 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volume_chop_regime_v2"
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
    
    # Choppiness Index (14-period)
    chop_period = 14
    atr_series = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - pd.Series(close).shift(1)), np.abs(low - pd.Series(close).shift(1)))))
    atr = atr_series.rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high_chop = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low_chop = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(atr / (highest_high_chop - lowest_low_chop)) / np.log10(chop_period)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(chop[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Choppiness regime filter: trending market (CHOP < 38.2)
        trending_regime = chop[i] < 38.2
        
        if position == 1:  # Long position
            # Exit: price reaches Donchian lower band or volume dries up or regime changes to choppy
            if close[i] <= lowest_low[i] or not volume_confirmed or not trending_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches Donchian upper band or volume dries up or regime changes to choppy
            if close[i] >= highest_high[i] or not volume_confirmed or not trending_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and trending_regime:
                # Long breakout: price breaks above Donchian upper band
                if close[i] > highest_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakdown: price breaks below Donchian lower band
                elif close[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals