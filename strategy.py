#!/usr/bin/env python3
# 6h_donchian_breakout_volume_regime_v1
# Hypothesis: 6h Donchian(20) breakout with volume confirmation and chop regime filter.
# Long: Price breaks above 20-period Donchian high + volume > 1.5x 20-period average + chop < 61.8 (trending regime).
# Short: Price breaks below 20-period Donchian low + volume > 1.5x 20-period average + chop < 61.8.
# Exit: Price returns to opposite Donchian level (exit long at Donchian low, exit short at Donchian high).
# Uses 6h primary timeframe with chop regime from same timeframe to avoid whipsaws in ranging markets.
# Designed for low trade frequency (~15-35/year) to minimize fee drag while capturing strong trends.
# Works in bull markets via breakouts and bear markets via breakdowns, with chop filter preventing entries in ranges.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Chop regime indicator (14-period) - uses same timeframe
    # Chop = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = high_s.rolling(window=14, min_periods=14).max().values
    min_low = low_s.rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = max_high - min_low
    sum_atr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    chop = np.zeros(n)
    for i in range(n):
        if range_14[i] > 0 and not np.isnan(sum_atr[i]) and not np.isnan(atr[i]):
            chop[i] = 100 * np.log10(sum_atr[i] / range_14[i]) / np.log10(14)
        else:
            chop[i] = 50  # neutral value when undefined
    
    # Chop threshold: < 61.8 = trending regime (good for breakouts)
    chop_threshold = 61.8
    trending_regime = chop < chop_threshold
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_ma[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to Donchian low
            if close[i] <= donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to Donchian high
            if close[i] >= donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Donchian breakout + volume + trending regime
            if (close[i] > donchian_high[i] and    # Break above Donchian high
                volume_confirmed and               # Volume spike
                trending_regime[i]):               # Trending regime (chop < 61.8)
                position = 1
                signals[i] = 0.25
            # Short entry: Donchian breakdown + volume + trending regime
            elif (close[i] < donchian_low[i] and   # Break below Donchian low
                  volume_confirmed and             # Volume spike
                  trending_regime[i]):             # Trending regime (chop < 61.8)
                position = -1
                signals[i] = -0.25
    
    return signals