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
    
    # Get 12h HTF data once before loop (primary HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Supertrend (ATR=10, mult=3) for trend filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range and ATR
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr_12h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2_12h = (high_12h + low_12h) / 2
    upper_12h = hl2_12h + 3 * atr_12h
    lower_12h = hl2_12h - 3 * atr_12h
    
    supertrend_12h = np.full_like(close_12h, np.nan)
    direction_12h = np.ones_like(close_12h)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_12h)):
        if np.isnan(supertrend_12h[i-1]):
            supertrend_12h[i] = upper_12h[i]
            direction_12h[i] = 1
        else:
            if close_12h[i] <= supertrend_12h[i-1]:
                direction_12h[i] = -1
            else:
                direction_12h[i] = 1
            
            if direction_12h[i] == 1:
                supertrend_12h[i] = max(lower_12h[i], supertrend_12h[i-1])
            else:
                supertrend_12h[i] = min(upper_12h[i], supertrend_12h[i-1])
    
    # Align 12h Supertrend direction to 4h
    supertrend_dir_4h = align_htf_to_ltf(prices, df_12h, direction_12h)
    
    # Get 1d HTF data for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3_1d = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d ATR ratio (current vs 50-period average) for regime
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio_1d = atr_1d / (atr_ma_50 + 1e-10)
    
    # Align 1d ATR ratio to 4h
    atr_ratio_4h = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate 4h Donchian channels (20-period) for entry signals
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 4h (no alignment needed, but keep for consistency)
    upper_20_4h = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_4h = align_htf_to_ltf(prices, df_4h, lower_20)
    
    # Calculate 4h ATR(14) for volatility filter
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr3_4h = np.abs(low_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    close_4h = df_4h['close'].values
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_4h_4h = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Calculate 4h volume ratio (current vs 20-period average)
    volume_4h = df_4h['volume'].values
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ratio_4h = volume_4h / (vol_ma_20 + 1e-10)
    volume_ratio_4h_4h = align_htf_to_ltf(prices, df_4h, volume_ratio_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_dir_4h[i]) or np.isnan(atr_ratio_4h[i]) or 
            np.isnan(upper_20_4h[i]) or np.isnan(lower_20_4h[i]) or 
            np.isnan(atr_4h_4h[i]) or np.isnan(volume_ratio_4h_4h[i]) or 
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when ATR ratio > 0.8 (avoid low volatility chop)
        if atr_ratio_4h[i] < 0.8:
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 12h Supertrend uptrend (direction = 1)
        # 2. 4h price breaks above 4h Donchian upper (20)
        # 3. Volume confirmation: volume > 1.3x average
        if (supertrend_dir_4h[i] == 1 and
            close[i] > upper_20_4h[i] and
            volume_ratio_4h_4h[i] > 1.3):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 12h Supertrend downtrend (direction = -1)
        # 2. 4h price breaks below 4h Donchian lower (20)
        # 3. Volume confirmation: volume > 1.3x average
        elif (supertrend_dir_4h[i] == -1 and
              close[i] < lower_20_4h[i] and
              volume_ratio_4h_4h[i] > 1.3):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_12h_Supertrend_Donchian20_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0