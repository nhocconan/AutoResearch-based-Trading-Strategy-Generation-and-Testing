#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v2
# Hypothesis: 4h Donchian(20) breakout with volume confirmation (>1.5x 20-period average) and chop regime filter (CHOP(14) > 61.8 for ranging market). 
# In ranging markets (2025+), price tends to revert from Donchian channel extremes. Volume filters false breakouts. 
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 75-200 total trades over 4 years by requiring breakout + volume + chop filter.
# Primary timeframe: 4h, HTF: 12h for regime filter to avoid look-ahead.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v2"
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
    
    # 12h HTF data for chop regime filter (to avoid look-ahead and ensure completed bar)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h choppiness index (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    atr_12h = pd.Series(high_12h - low_12h).rolling(window=14, min_periods=14).sum().values
    high_14_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    low_14_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero in chop formula
    chop_denom = np.log10(atr_12h) * np.log10(14)
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop_12h = 100 * np.log10((high_14_12h - low_14_12h) / chop_denom) / np.log10(14)
    
    # Align 12h chop to 4h timeframe (completed-bar timing)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Donchian channel (20-period) on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Chop regime: only trade when market is ranging (chop > 61.8)
        chop_regime = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price moves below Donchian lower band or volume dries up
            if close[i] < low_20[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above Donchian upper band or volume dries up
            if close[i] > high_20[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and chop_regime:
                # Long entry: price breaks above Donchian upper band with volume
                if close[i] > high_20[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian lower band with volume
                elif close[i] < low_20[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals