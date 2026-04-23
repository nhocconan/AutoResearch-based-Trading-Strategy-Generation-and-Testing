#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 1d ATR filter and volume confirmation.
Long when price breaks above Donchian(20) upper band AND 1d ATR ratio > 1.2 (low volatility expansion) with volume > 1.5x average.
Short when price breaks below Donchian(20) lower band AND 1d ATR ratio > 1.2 with volume > 1.5x average.
Exit when price breaks opposite Donchian band or ATR ratio < 0.8 (volatility contraction).
Uses 4h timeframe to target 75-200 trades over 4 years. Donchian provides clear structure, ATR filter ensures breakouts occur after low volatility (volatility contraction -> expansion), volume confirms conviction.
Works in both bull and bear markets by capturing volatility breakouts regardless of direction.
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
    
    # Load 1d data for ATR calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) and ATR ratio (current ATR / 20-period MA of ATR)
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # First period
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio_1d = atr_1d / atr_ma_1d
    
    # Align HTF indicators to 4h timeframe
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate Donchian channels on 4h (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr_ratio_val = atr_ratio_1d_aligned[i]
        upper_band = high_roll[i]
        lower_band = low_roll[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND ATR ratio > 1.2 (vol expansion) AND volume confirmation
            if (price > upper_band and atr_ratio_val > 1.2 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND ATR ratio > 1.2 AND volume confirmation
            elif (price < lower_band and atr_ratio_val > 1.2 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian lower band OR ATR ratio < 0.8 (vol contraction)
                if price < lower_band or atr_ratio_val < 0.8:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Donchian upper band OR ATR ratio < 0.8
                if price > upper_band or atr_ratio_val < 0.8:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dATR_Ratio_Volume"
timeframe = "4h"
leverage = 1.0