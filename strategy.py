#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian breakout with volume confirmation
# Choppiness Index > 61.8 indicates ranging market (mean reversion)
# Choppiness Index < 38.2 indicates trending market (trend following)
# In ranging markets: buy near Donchian lower band, sell near upper band
# In trending markets: buy breakout above upper band, sell breakout below lower band
# Volume confirmation: current volume > 1.5x 20-period average
# Designed to work in both trending and ranging markets
# Target: 20-40 trades/year to avoid fee drag
name = "4h_Chop_Donchian_Breakout_Volume_v1"
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
    
    # Get 1d data for Choppiness Index (trend regime filter)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Choppiness Index on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # ATR(14) for daily
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    # Avoid division by zero
    range_14 = highest_high_14 - lowest_low_14
    chop_1d = np.full_like(close_1d, 50.0)  # default to neutral
    mask = range_14 > 0
    chop_1d[mask] = 100 * np.log10(sum_atr_14[mask] / range_14[mask]) / np.log10(14)
    
    # Align Choppiness Index to 4h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # 4h Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h ATR for stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(chop_1d_aligned[i]) or \
           np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or np.isnan(atr_4h[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.5 * avg_volume
        
        chop = chop_1d_aligned[i]
        donchian_upper = highest_high_20[i]
        donchian_lower = lowest_low_20[i]
        
        if position == 0:
            # Ranging market (chop > 61.8): mean reversion at extremes
            if chop > 61.8:
                # Buy near lower band, sell near upper band
                if close[i] <= donchian_lower * 1.001 and volume_filter:  # slight buffer
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= donchian_upper * 0.999 and volume_filter:  # slight buffer
                    signals[i] = -0.25
                    position = -1
            # Trending market (chop < 38.2): breakout in direction of trend
            elif chop < 38.2:
                # Buy breakout above upper band
                if close[i] > donchian_upper and volume_filter:
                    signals[i] = 0.25
                    position = 1
                # Sell breakout below lower band
                elif close[i] < donchian_lower and volume_filter:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit conditions
            exit_signal = False
            # In ranging market: exit when price reaches opposite band
            if chop > 61.8 and close[i] >= donchian_upper * 0.999:
                exit_signal = True
            # In trending market: exit on breakdown below lower band
            elif chop < 38.2 and close[i] < donchian_lower:
                exit_signal = True
            # Always exit on 2x ATR stop
            elif close[i] < highest_high_20[i] - 2.0 * atr_4h[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions
            exit_signal = False
            # In ranging market: exit when price reaches opposite band
            if chop > 61.8 and close[i] <= donchian_lower * 1.001:
                exit_signal = True
            # In trending market: exit on breakout above upper band
            elif chop < 38.2 and close[i] > donchian_upper:
                exit_signal = True
            # Always exit on 2x ATR stop
            elif close[i] > lowest_low_20[i] + 2.0 * atr_4h[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals