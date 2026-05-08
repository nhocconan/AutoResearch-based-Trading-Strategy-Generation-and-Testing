#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index + Donchian breakout with volume confirmation.
# Long when Choppiness < 38.2 (trending) AND price breaks above Donchian upper band (20) AND volume > 1.3x 20-period average.
# Short when Choppiness < 38.2 (trending) AND price breaks below Donchian lower band (20) AND volume > 1.3x 20-period average.
# Exit when Choppiness > 61.8 (ranging) OR price crosses back inside Donchian channel.
# This strategy combines trend detection (Choppiness) with breakout logic to avoid false signals in ranging markets.
# Works in both bull and bear by adapting to trending conditions only, avoiding choppy periods where breakouts fail.

name = "4h_Chop_Donchian_Breakout_Volume"
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
    
    # 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 1d data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period high/low) from 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper band (20-period high)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian lower band (20-period low)
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (actually same timeframe, but keeping for consistency)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate Choppiness Index (14-period) from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index formula
    chop = np.zeros_like(close_1d)
    mask = (hh14 - ll14) != 0
    chop[mask] = 100 * np.log10(sum_tr14[mask] / (hh14[mask] - ll14[mask])) / np.log10(14)
    chop = np.where((hh14 - ll14) == 0, 50, chop)  # Neutral when no range
    
    # Align Choppiness to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: trending market (Chop < 38.2) AND breakout above Donchian high AND volume filter
            long_cond = (chop_aligned[i] < 38.2) and (close[i] > donchian_high_aligned[i]) and volume_filter[i]
            # Short conditions: trending market (Chop < 38.2) AND breakdown below Donchian low AND volume filter
            short_cond = (chop_aligned[i] < 38.2) and (close[i] < donchian_low_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: ranging market (Chop > 61.8) OR price crosses back below Donchian low
            if (chop_aligned[i] > 61.8) or (close[i] < donchian_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: ranging market (Chop > 61.8) OR price crosses back above Donchian high
            if (chop_aligned[i] > 61.8) or (close[i] > donchian_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals