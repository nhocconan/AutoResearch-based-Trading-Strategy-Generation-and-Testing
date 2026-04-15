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
    
    # Get 1d HTF data once before loop for all HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) and EMA(200) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMAs to 4h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 4h Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 4h (no shift needed as we're already on 4h timeframe)
    upper_20_4h = upper_20
    lower_20_4h = lower_20
    
    # Calculate 4h ATR(14) for volatility filter and position sizing
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.concatenate([[close_4h[0]], close_4h[:-1]])) if len(close_4h) > 1 else np.array([np.inf])
    tr3 = np.abs(low_4h - np.concatenate([[close_4h[0]], close_4h[:-1]])) if len(close_4h) > 1 else np.array([np.inf])
    close_4h = df_4h['close'].values
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 4h ATR to 4h
    atr_14_4h = atr_14
    
    # Calculate 4h volume ratio (current vs 20-period average)
    volume_4h = df_4h['volume'].values
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume_4h / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # For 4h timeframe, we work directly with 4h bars
    for i in range(50, len(prices)):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_4h[i]) or np.isnan(lower_20_4h[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(atr_14_4h[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Get 4h price data
        idx_4h = i  # Since we're on 4h timeframe, index maps directly
        if idx_4h >= len(close_4h):
            signals[i] = 0.0
            continue
            
        c = close_4h[idx_4h]
        
        # Long conditions:
        # 1. 4h price breaks above 4h Donchian upper (20)
        # 2. 1d EMA(50) > 1d EMA(200) (bullish trend)
        # 3. Volume confirmation: volume > 1.5x average
        if (c > upper_20_4h[idx_4h] and
            ema_50_1d_aligned[i] > ema_200_1d_aligned[i] and
            volume_ratio[idx_4h] > 1.5):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 4h price breaks below 4h Donchian lower (20)
        # 2. 1d EMA(50) < 1d EMA(200) (bearish trend)
        # 3. Volume confirmation: volume > 1.5x average
        elif (c < lower_20_4h[idx_4h] and
              ema_50_1d_aligned[i] < ema_200_1d_aligned[i] and
              volume_ratio[idx_4h] > 1.5):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1d_EMA50_200_Volume_Filter"
timeframe = "4h"
leverage = 1.0