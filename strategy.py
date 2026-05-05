#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and choppiness regime filter
# Long when: Price breaks above Donchian upper (20) AND 1d volume > 1.5x 20-day avg AND choppiness > 61.8 (range regime)
# Short when: Price breaks below Donchian lower (20) AND 1d volume > 1.5x 20-day avg AND choppiness > 61.8 (range regime)
# Exit when price returns to Donchian midpoint (mean reversion in range)
# Donchian breakout captures volatility expansion after consolidation in ranging markets
# Volume spike confirms institutional interest in the breakout
# Choppiness regime filter ensures we only trade in ranging markets (avoid strong trends where breakouts fail)
# Works in both bull and bear markets by trading mean-reverting breakouts in ranges
# Target: 100-180 total trades over 4 years (25-45/year) with discrete sizing 0.30

name = "4h_Donchian20_VolumeSpike_ChopRegime"
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
    
    # Get 1d data ONCE before loop for volume and choppiness filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for calculations
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d average volume (20-day)
    avg_volume_20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * avg_volume_20d)
    
    # Calculate 1d choppiness index (14-period)
    def true_range(high, low, close_prev):
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    close_prev_1d = np.roll(close_1d, 1)
    close_prev_1d[0] = np.nan
    tr_1d = true_range(high_1d, low_1d, close_prev_1d)
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    max_high_14_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    denominator = max_high_14_1d - min_low_14_1d
    chop_1d = np.where(denominator != 0, 
                       100 * np.log10(atr_14_1d / denominator * np.sqrt(14)) / np.log10(14), 
                       50.0)  # neutral value when no range
    
    chop_regime = chop_1d > 61.8  # ranging market
    
    # Align 1d filters to 4h
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime.astype(float))
    
    # Calculate Donchian channels (20-period) on 4h
    upper_dc = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_dc = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_dc = (upper_dc + lower_dc) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(volume_spike_aligned[i]) or np.isnan(chop_regime_aligned[i]) or 
            np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or np.isnan(mid_dc[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper Donchian in ranging market with volume spike
            if close[i] > upper_dc[i] and chop_regime_aligned[i] > 0.5 and volume_spike_aligned[i] > 0.5:
                signals[i] = 0.30
                position = 1
            # Short: Break below lower Donchian in ranging market with volume spike
            elif close[i] < lower_dc[i] and chop_regime_aligned[i] > 0.5 and volume_spike_aligned[i] > 0.5:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: return to Donchian midpoint (mean reversion)
            if close[i] < mid_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: return to Donchian midpoint (mean reversion)
            if close[i] > mid_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals