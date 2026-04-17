#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d ATR-based volatility expansion filter to avoid whipsaws.
Long when price breaks above 1d Donchian(20) high with volume > 1.5x 20-period average and ATR(14) > 1.3x ATR(50).
Short when price breaks below 1d Donchian(20) low with same filters.
Uses 1d structure for institutional reference points, volatility expansion to confirm genuine breakouts
(not low-volatility squeezes), and volume for participation. Designed to capture strong trending moves
while avoiding false breakouts in ranging markets. Works in bull (breakouts in uptrend) and bear (breakdowns in downtrend)
by requiring volatility expansion as a regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian(20) channels
    high_ma_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR(14) and ATR(50) for volatility expansion filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # first TR is NaN
    
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr1).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 6h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe
    high_donchian_aligned = align_htf_to_ltf(prices, df_1d, high_ma_20)
    low_donchian_aligned = align_htf_to_ltf(prices, df_1d, low_ma_20)
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    atr50_aligned = align_htf_to_ltf(prices, df_1d, atr50)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for ATR50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_donchian_aligned[i]) or 
            np.isnan(low_donchian_aligned[i]) or 
            np.isnan(atr14_aligned[i]) or 
            np.isnan(atr50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility expansion: ATR(14) > 1.3x ATR(50) indicates expanding volatility
        vol_expansion = atr14_aligned[i] > 1.3 * atr50_aligned[i]
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 1d Donchian high with vol expansion and volume
            if (close[i] > high_donchian_aligned[i] and 
                vol_expansion and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian low with vol expansion and volume
            elif (close[i] < low_donchian_aligned[i] and 
                  vol_expansion and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 1d Donchian low (opposite side)
            if close[i] < low_donchian_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 1d Donchian high (opposite side)
            if close[i] > high_donchian_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dDonchian20_VolExpansion_Volume_Confirm"
timeframe = "6h"
leverage = 1.0