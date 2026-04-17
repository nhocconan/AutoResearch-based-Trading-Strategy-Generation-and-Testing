#!/usr/bin/env python3
"""
4h Williams Alligator with Daily Volume Spike and Chop Filter
Long: Price above Alligator teeth (middle line) + volume spike + chop > 61.8
Short: Price below Alligator teeth + volume spike + chop > 61.8
Exit: Price crosses Alligator teeth in opposite direction
Williams Alligator identifies trend; volume spike confirms momentum; chop filter ensures trending regime.
Designed to work in both bull and bear markets by capturing strong trending moves.
Target: 80-160 total trades over 4 years (20-40/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_alligator(high, low, close):
    """Calculate Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs with future shift"""
    # Jaw: 13-period SMMA shifted 8 bars forward
    sma13 = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = sma13.shift(8)
    
    # Teeth: 8-period SMMA shifted 5 bars forward
    sma8 = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = sma8.shift(5)
    
    # Lips: 5-period SMMA shifted 3 bars forward
    sma5 = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = sma5.shift(3)
    
    return jaw, teeth, lips

def calculate_chop(high, low, close, period=14):
    """Calculate Choppiness Index: higher = ranging, lower = trending"""
    atr = pd.Series(np.sqrt((high - low)**2)).rolling(window=period, min_periods=period).sum()
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    chop = 100 * np.log10(atr / (highest_high - lowest_low)) / np.log10(period)
    return chop.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 4h
    jaw, teeth, lips = calculate_alligator(high, low, close)
    
    # Get 1d data for volume average and chop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d average volume (20-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 1d Choppiness Index
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 30  # need Alligator calculations
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_1d_aligned[i] if vol_ma_1d_aligned[i] > 0 else 0
        chop_value = chop_1d_aligned[i]
        
        # Volume spike: at least 1.5x average daily volume
        volume_spike = vol_ratio >= 1.5
        # Trending regime: chop > 61.8 indicates ranging, so we want chop <= 61.8 for trending
        trending_regime = chop_value <= 61.8
        
        if position == 0:
            # Long: price above teeth + volume spike + trending regime
            if price > teeth[i] and volume_spike and trending_regime:
                signals[i] = 0.25
                position = 1
            # Short: price below teeth + volume spike + trending regime
            elif price < teeth[i] and volume_spike and trending_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below teeth
            if price < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above teeth
            if price > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_VolumeSpike_Chop"
timeframe = "4h"
leverage = 1.0