#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 4h (same timeframe, just for completed bar alignment)
    upper_20_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_20_4h)
    lower_20_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_20_4h)
    
    # Get 1d HTF data for ATR-based volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d ATR ratio (current vs 50-period average) for regime filter
    atr_ma_50_1d = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio_1d = atr_14_1d / (atr_ma_50_1d + 1e-10)
    
    # Align 1d ATR ratio to 4h
    atr_ratio_4h_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_4h_aligned[i]) or np.isnan(lower_20_4h_aligned[i]) or 
            np.isnan(atr_ratio_4h_aligned[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 4h price breaks above 4h Donchian upper (20) - bullish breakout
        # 2. High volatility regime: ATR ratio > 1.2 (avoid low volatility chop)
        # 3. Volume confirmation: volume > 1.5x average
        if (close[i] > upper_20_4h_aligned[i] and
            atr_ratio_4h_aligned[i] > 1.2 and
            volume_ratio[i] > 1.5):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 4h price breaks below 4h Donchian lower (20) - bearish breakdown
        # 2. High volatility regime: ATR ratio > 1.2
        # 3. Volume confirmation: volume > 1.5x average
        elif (close[i] < lower_20_4h_aligned[i] and
              atr_ratio_4h_aligned[i] > 1.2 and
              volume_ratio[i] > 1.5):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Volume_ATRRegime_Filter_v1"
timeframe = "4h"
leverage = 1.0