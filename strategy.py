#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation
# Long when: price breaks above 20-period Donchian high, 1d ATR ratio > 1.2 (expanding volatility), volume > 1.5x 20-bar average
# Short when: price breaks below 20-period Donchian low, 1d ATR ratio > 1.2, volume > 1.5x 20-bar average
# Exit when price returns to midpoint of Donchian channel or opposite breakout
# Works in bull markets (breakout continuation) and bear markets (volatility expansion captures panic moves)
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_Breakout_1dATR_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) and ATR ratio (current ATR / 20-period MA of ATR)
    if len(high_1d) >= 14:
        # True Range components
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        # Prepend first TR as high-low
        tr = np.concatenate([[high_1d[0] - low_1d[0]], tr])
        
        # ATR(14)
        atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        # 20-period MA of ATR
        atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
        # ATR ratio: current ATR / 20-period MA of ATR
        atr_ratio = np.where(atr_ma_20 > 0, atr_14 / atr_ma_20, 1.0)
    else:
        atr_ratio = np.ones(len(close_1d))
    
    # Align ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 4h Donchian channels (20-period)
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, expanding volatility, volume confirmation
            if (close[i] > donchian_high[i] and 
                atr_ratio_aligned[i] > 1.2 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low, expanding volatility, volume confirmation
            elif (close[i] < donchian_low[i] and 
                  atr_ratio_aligned[i] > 1.2 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian midpoint or breaks below Donchian low
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian midpoint or breaks above Donchian high
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals