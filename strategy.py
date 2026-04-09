#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_regime_v3
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and choppiness regime filter.
# In trending markets (chop < 38.2), breakouts capture momentum; in ranging markets (chop > 61.8),
# faded breakouts mean-revert. Volume confirmation filters false breakouts.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 75-200 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_regime_v3"
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
    
    # 12h HTF data for regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness index regime filter (14-period) from 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    atr_14 = pd.Series(high_12h - low_12h).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    chop_denom = np.log10(atr_14) * np.log10(14)
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10((high_14 - low_14) / chop_denom) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Chop regime: trending (chop < 38.2) or ranging (chop > 61.8)
        chop_trending = chop_aligned[i] < 38.2
        chop_ranging = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price moves below Donchian low or volume dries up
            if close[i] < low_20[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above Donchian high or volume dries up
            if close[i] > high_20[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                if chop_trending:
                    # Trending market: breakout follow-through
                    if close[i] > high_20[i]:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] < low_20[i]:
                        position = -1
                        signals[i] = -0.25
                elif chop_ranging:
                    # Ranging market: faded breakout mean reversion
                    if close[i] > high_20[i]:
                        position = -1  # short failed breakout
                        signals[i] = -0.25
                    elif close[i] < low_20[i]:
                        position = 1   # long failed breakdown
                        signals[i] = 0.25
    
    return signals