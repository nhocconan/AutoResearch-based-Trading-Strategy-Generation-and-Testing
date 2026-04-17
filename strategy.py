#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1_S1_Breakout_VolumeRegime_V1
Breakout at Camarilla R1/S1 levels derived from 1d high/low/close, with 1d volume confirmation and 1d Choppiness regime filter.
Long when price breaks above R1 with volume > 1.5x 20-period MA and CHOP > 61.8 (range market).
Short when price breaks below S1 with volume > 1.5x 20-period MA and CHOP > 61.8.
Exit when price crosses back below/above the 1d close (pivot point).
Position size: 0.25. Target: 20-40 trades/year.
Works in bull/bear: mean reversion in range (CHOP > 61.8), avoids trending markets where false breakouts occur.
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
    
    # Get 1D data once
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels for each 1D bar
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    # Pivot = close
    hl_range = high_1d - low_1d
    r1 = close_1d + hl_range * 1.1 / 12
    s1 = close_1d - hl_range * 1.1 / 12
    pivot = close_1d
    
    # Align Camarilla levels to 4H
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    
    # 1D volume filter: volume > 1.5x 20-period MA
    volume_ma20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma20_4h = align_htf_to_ltf(prices, df_1d, volume_ma20)
    
    # 1D Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate CHOP over 14 periods
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    max_high14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (max_high14 - min_low14 + 1e-10)) / np.log10(14)
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # warmup for rolling calculations
        # Skip if any required data is not available
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(pivot_4h[i]) or
            np.isnan(volume_ma20_4h[i]) or np.isnan(chop_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume and regime filters
        volume_filter = volume[i] > (1.5 * volume_ma20_4h[i])
        regime_filter = chop_4h[i] > 61.8  # Range market (CHOP > 61.8)
        
        if position == 0:
            # Long when price breaks above R1 with volume and range regime
            if close[i] > r1_4h[i] and volume_filter and regime_filter:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S1 with volume and range regime
            elif close[i] < s1_4h[i] and volume_filter and regime_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price crosses below pivot (1D close)
            if close[i] < pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price crosses above pivot (1D close)
            if close[i] > pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1_S1_Breakout_VolumeRegime_V1"
timeframe = "4h"
leverage = 1.0