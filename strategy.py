#!/usr/bin/env python3
# 4h_donchian_1d_chop_volume_v1
# Hypothesis: 4h Donchian(20) breakout with 1d choppiness filter and volume confirmation.
# Long: price breaks above Donchian(20) high + chop(1d) > 61.8 (range) + volume > 1.5x 20MA
# Short: price breaks below Donchian(20) low + chop(1d) > 61.8 + volume > 1.5x 20MA
# Uses discrete sizing (±0.25) to minimize fee churn. Target: 75-200 trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_1d_chop_volume_v1"
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
    
    # 1d HTF data for choppiness index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14) for 1d
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index: 100 * log10(sum(ATR14) / (max(highN) - min(lowN))) / log10(N)
    # where N=14
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((max_high - min_low) == 0, 50, chop)
    chop = np.where(np.isnan(chop), 50, chop)
    
    # Align choppiness to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian(20) on 4h
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Range regime: chop > 61.8 indicates ranging market (good for mean reversion/breakouts)
        in_range = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR chop drops below 38.2 (trending)
            if close[i] < donch_low[i] or chop_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR chop drops below 38.2 (trending)
            if close[i] > donch_high[i] or chop_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation and ranging regime
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if in_range and volume_confirmed:
                # Long: price breaks above Donchian high
                if close[i] > donch_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low
                elif close[i] < donch_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals