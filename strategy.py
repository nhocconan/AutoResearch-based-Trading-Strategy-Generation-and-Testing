#!/usr/bin/env python3
# 12h_Price_Channel_Breakout_Volume_Regime
# Hypothesis: Use 12h Donchian channel breakout with volume confirmation and 1d Choppiness regime filter.
# Long when price breaks above upper Donchian (20) with volume spike and chop < 61.8 (trending).
# Short when price breaks below lower Donchian (20) with volume spike and chop < 61.8.
# Exit on opposite breakout or when chop > 61.8 (range) to avoid whipsaws.
# Designed for low frequency (15-30 trades/year) to avoid fee drag. Works in bull/bear via trend filter.

name = "12h_Price_Channel_Breakout_Volume_Regime"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def donchian_channels(high, low, lookback=20):
    """Calculate Donchian channels: upper = max(high, lookback), lower = min(low, lookback)."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(n):
        if i >= lookback - 1:
            upper[i] = np.max(high[i - lookback + 1:i + 1])
            lower[i] = np.min(low[i - lookback + 1:i + 1])
    
    return upper, lower

def choppiness_index(high, low, close, period=14):
    """Calculate Choppiness Index: higher = ranging, lower = trending."""
    n = len(high)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    atr = np.zeros(n)
    for i in range(n):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        if i < period:
            atr[i] = np.mean(atr[:i+1]) if i > 0 else tr
        else:
            atr[i] = (atr[i-1] * (period-1) + tr) / period
    
    for i in range(period-1, n):
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        if hh == ll:
            chop[i] = 50.0
        else:
            chop[i] = 100 * np.log10(np.sum(atr[i-period+1:i+1]) / np.log(period) / (hh - ll))
    
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Choppiness Index filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    upper, lower = donchian_channels(high, low, 20)
    
    # Calculate average volume (20-period) for spike detection
    vol_ma = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # Calculate 1d Choppiness Index
    chop_1d = choppiness_index(high_1d, low_1d, close_1d, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure Donchian and volume MA are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 1.5 * 20-period average
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Trending regime: Choppiness Index < 61.8
        trending_regime = chop_1d_aligned[i] < 61.8
        
        if position == 0:
            # LONG: Price breaks above upper Donchian + volume spike + trending regime
            if close[i] > upper[i] and volume_spike and trending_regime:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + volume spike + trending regime
            elif close[i] < lower[i] and volume_spike and trending_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian OR chop > 61.8 (range)
            if close[i] < lower[i] or chop_1d_aligned[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian OR chop > 61.8 (range)
            if close[i] > upper[i] or chop_1d_aligned[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals