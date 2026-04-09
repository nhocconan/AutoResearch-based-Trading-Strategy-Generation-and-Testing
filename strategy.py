#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume confirmation + ATR filter
# - Primary signal: 12h price break above Donchian upper(20) for long, below Donchian lower(20) for short
# - Volume filter: 1d volume > 20-period median volume (ensures participation)
# - ATR filter: 12h ATR(14) > 0.5 * 20-period median ATR (avoid low-volatility chop)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Donchian breakouts capture strong moves, volume/ATR filters avoid false signals in ranging markets

name = "12h_1d_donchian_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d volume median
    volume_1d = df_1d['volume'].values
    median_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    volume_median_aligned = align_htf_to_ltf(prices, df_1d, median_volume_20)
    
    # Pre-compute 12h Donchian channels
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Donchian upper/lower (20-period)
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 12h ATR(14)
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar TR = high - low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 12h ATR median
    median_atr_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(volume_median_aligned[i]) or
            np.isnan(atr_14[i]) or
            np.isnan(median_atr_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume and ATR filters
        volume_ok = prices['volume'].iloc[i] > volume_median_aligned[i]
        atr_ok = atr_14[i] > 0.5 * median_atr_20[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower(20) OR ATR drops too low
            if close_12h[i] < donchian_lower[i] or not atr_ok:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper(20) OR ATR drops too low
            if close_12h[i] > donchian_upper[i] or not atr_ok:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian upper(20) with volume and ATR confirmation
            if (close_12h[i] > donchian_upper[i] and 
                volume_ok and 
                atr_ok):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian lower(20) with volume and ATR confirmation
            elif (close_12h[i] < donchian_lower[i] and 
                  volume_ok and 
                  atr_ok):
                position = -1
                signals[i] = -0.25
    
    return signals