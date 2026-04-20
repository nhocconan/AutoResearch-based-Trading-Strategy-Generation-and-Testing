#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for HTF analysis (ATR for regime filter)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 14-period ATR on 1d for regime filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr_14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # 14-period ATR average (200-day) for regime threshold
    atr_ma_200 = pd.Series(atr_14).rolling(window=200, min_periods=200).mean().values
    atr_ratio = atr_14 / atr_ma_200  # >1 = high volatility regime
    
    # Align ATR ratio to 12h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Load 12h data for entry logic
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Donchian channels (20-period) on 12h
    highest_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection (20-period on 12h)
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        vol = volume_12h[i]
        
        # Regime filter: only trade in high volatility (ATR ratio > 1.2)
        in_high_vol = atr_ratio_aligned[i] > 1.2
        
        if position == 0 and in_high_vol:
            # Long: break above Donchian high with volume confirmation
            if price > highest_20[i] and vol > 1.5 * vol_ma_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume confirmation
            elif price < lowest_20[i] and vol > 1.5 * vol_ma_20[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below Donchian low
            if price < lowest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above Donchian high
            if price > highest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_Volume_ATRRegime_v1"
timeframe = "12h"
leverage = 1.0