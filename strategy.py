#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Squeeze + 1d Volume Spike + Choppiness Regime Filter
# Bollinger Band squeeze identifies low volatility periods. When volatility expands (band width > 20-day average)
# and is confirmed by volume spike (>2x 20-period average) and trending market (Choppiness Index < 38.2),
# we enter in the direction of the breakout. Works in both bull and bear markets by capturing volatility expansion.
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 4h
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = upper_bb - lower_bb
    
    # Bollinger Band squeeze: BB width < 20-period average of BB width
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Volatility expansion: BB width > 1.5x 20-period average
    volatility_expansion = bb_width > 1.5 * bb_width_ma
    
    # Volume spike: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2 * volume_ma
    
    # Load 1d data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for Chop
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of true ranges over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(atr_sum / (highest_high - lowest_low)) / log10(14)
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(14)
    
    # Align Chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Trending market: Chop < 38.2
    trending = chop_aligned < 38.2
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(chop_aligned[i])):
            continue
        
        # Long entry: volatility expansion + volume spike + trending + close above upper BB
        if (volatility_expansion[i] and volume_spike[i] and trending[i] and
            close[i] > upper_bb[i] and position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: volatility expansion + volume spike + trending + close below lower BB
        elif (volatility_expansion[i] and volume_spike[i] and trending[i] and
              close[i] < lower_bb[i] and position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: volatility contraction (squeeze) or opposite BB touch
        elif position == 1 and (squeeze[i] or close[i] < lower_bb[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (squeeze[i] or close[i] > upper_bb[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Bollinger_Squeeze_Volume_Chop"
timeframe = "4h"
leverage = 1.0