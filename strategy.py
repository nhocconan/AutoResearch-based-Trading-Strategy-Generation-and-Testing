#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian channel breakout with 1d ATR filter and volume confirmation.
Long when price breaks above 12h Donchian upper (20) AND 1d ATR(14) > 1.2x 20-period MA AND volume > 1.5x average.
Short when price breaks below 12h Donchian lower (20) AND 1d ATR(14) > 1.2x 20-period MA AND volume > 1.5x average.
Exit when price reverts to 12h Donchian middle line or volatility collapses (ATR ratio < 0.8).
Uses 12h timeframe to target ~15-30 trades/year, minimizing fee drag while capturing volatility expansion breakouts.
Works in both bull and bear markets by requiring volatility expansion (high ATR) for breakout entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ATR filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr1 = np.concatenate([[np.nan], tr1])
    tr2 = np.concatenate([[np.nan], tr2])
    tr3 = np.concatenate([[np.nan], tr3])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR 20-period MA for volatility regime filter
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ma_20_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20)
    
    # Calculate 12h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_20 = (highest_20 + lowest_20) / 2
    
    # Volume average (20-period) on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(middle_20[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_ma_20_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_20 = highest_20[i]
        lower_20 = lowest_20[i]
        middle_20_val = middle_20[i]
        atr_14_val = atr_14_aligned[i]
        atr_ma_20_val = atr_ma_20_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        # Volatility filter: ATR > 1.2x MA (expanding volatility)
        vol_expanding = atr_14_val > 1.2 * atr_ma_20_val
        # Volume confirmation: current volume > 1.5x average
        volume_spike = vol_current > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above Donchian upper AND volatility expanding AND volume spike
            if price > upper_20 and vol_expanding and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND volatility expanding AND volume spike
            elif price < lower_20 and vol_expanding and volume_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to middle line OR volatility collapses (ATR < 0.8x MA)
                if price < middle_20_val or atr_14_val < 0.8 * atr_ma_20_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to middle line OR volatility collapses (ATR < 0.8x MA)
                if price > middle_20_val or atr_14_val < 0.8 * atr_ma_20_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1dATR_Volume_Breakout"
timeframe = "12h"
leverage = 1.0