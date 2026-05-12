#!/usr/bin/env python3
name = "4h_Chop_Adapted_Donchian_Breakout"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D DATA FOR REGIME FILTER (CHOP) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Chop Index (14-period)
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First value
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(atr14.sum() / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop = np.where((highest_high_14 - lowest_low_14) != 0, chop, 50)  # Avoid division by zero
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 4H DATA FOR DONCHIAN BREAKOUT ===
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(chop_4h[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Regime filter: Chop > 61.8 = ranging (mean revert), Chop < 38.2 = trending (trend follow)
        is_ranging = chop_4h[i] > 61.8
        is_trending = chop_4h[i] < 38.2
        
        if position == 0:
            # In ranging market: mean reversion at Donchian extremes
            if is_ranging:
                if low[i] <= lowest_low_20[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif high[i] >= highest_high_20[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            # In trending market: breakout in direction of trend
            elif is_trending:
                if high[i] > highest_high_20[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                elif low[i] < lowest_low_20[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price reaches opposite Donchian band or volatility drops
            if high[i] >= highest_high_20[i] or chop_4h[i] > 50:  # Exit at upper band or when ranging
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches opposite Donchian band or volatility drops
            if low[i] <= lowest_low_20[i] or chop_4h[i] > 50:  # Exit at lower band or when ranging
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals