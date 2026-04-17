#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation.
# Uses 20-period Donchian channels from 4h data for breakout detection.
# Adds 1d ATR(14) filter: only trade when volatility is above average (avoid chop).
# Volume confirmation: current volume > 1.5 * 20-period average.
# Designed for fewer, higher-quality trades (target: 20-50/year) to minimize fee drag.
# Works in trending markets (breakouts) and avoids ranging markets via volatility filter.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_max_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (already aligned, but keep for consistency)
    donchian_high_4h = align_htf_to_ltf(prices, df_4h, high_max_20)
    donchian_low_4h = align_htf_to_ltf(prices, df_4h, low_min_20)
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR(50) for average volatility reference
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align ATR values to 4h timeframe
    atr_14_4h = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_4h = align_htf_to_ltf(prices, df_1d, atr_50)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for ATR and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_4h[i]) or 
            np.isnan(donchian_low_4h[i]) or 
            np.isnan(atr_14_4h[i]) or 
            np.isnan(atr_50_4h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when current ATR > average ATR
        volatility_filter = atr_14_4h[i] > atr_50_4h[i]
        
        # Volume filter: spike > 1.5x average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Price relative to Donchian channels
        price_above_high = close[i] > donchian_high_4h[i]
        price_below_low = close[i] < donchian_low_4h[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high with volatility and volume
            if (price_above_high and volatility_filter and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volatility and volume
            elif (price_below_low and volatility_filter and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below Donchian low
            if close[i] < donchian_low_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above Donchian high
            if close[i] > donchian_high_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATR_Volume_Filter"
timeframe = "4h"
leverage = 1.0