#!/usr/bin/env python3
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 50-period EMA on weekly close
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate 20-period ATR on weekly for volatility filter
    tr_1w = np.maximum(high_1w[1:] - low_1w[1:], 
                       np.maximum(np.abs(high_1w[1:] - close_1w[:-1]), 
                                  np.abs(low_1w[1:] - close_1w[:-1])))
    tr_1w = np.concatenate([[np.nan], tr_1w])
    atr_1w = np.full(len(tr_1w), np.nan)
    for i in range(20, len(tr_1w)):
        if i == 20:
            atr_1w[i] = np.mean(tr_1w[1:21])
        else:
            atr_1w[i] = (atr_1w[i-1] * 19 + tr_1w[i]) / 20
    
    # Align weekly indicators to 12h
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Get daily data for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channels on daily
    upper_20 = np.full(len(high_1d), np.nan)
    lower_20 = np.full(len(low_1d), np.nan)
    for i in range(19, len(high_1d)):
        upper_20[i] = np.max(high_1d[i-19:i+1])
        lower_20[i] = np.min(low_1d[i-19:i+1])
    
    # Align Donchian to 12h
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly EMA/ATR, daily Donchian, and volume MA
    start_idx = max(50, 20, 19, vol_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_1w_aligned[i]) or 
            np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        atr = atr_1w_aligned[i]
        
        if position == 0:
            # Long: Price breaks above weekly EMA50 and Donchian upper with volume
            if (price > ema_50_1w_aligned[i] and 
                price > upper_20_aligned[i] and 
                vol_ratio > 1.8):
                signals[i] = size
                position = 1
            # Short: Price breaks below weekly EMA50 and Donchian lower with volume
            elif (price < ema_50_1w_aligned[i] and 
                  price < lower_20_aligned[i] and 
                  vol_ratio > 1.8):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below weekly EMA50 or Donchian lower or ATR stop
            if (price < ema_50_1w_aligned[i] or 
                price < lower_20_aligned[i] or 
                price < close[i-1] - 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above weekly EMA50 or Donchian upper or ATR stop
            if (price > ema_50_1w_aligned[i] or 
                price > upper_20_aligned[i] or 
                price > close[i-1] + 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_EMA50_Donchian20_VolumeFilter"
timeframe = "12h"
leverage = 1.0