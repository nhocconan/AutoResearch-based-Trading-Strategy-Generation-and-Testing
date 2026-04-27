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
    
    # Get 1d data for ATR and Close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR (14)
    atr_period = 14
    atr = np.full(len(close_1d), np.nan)
    if len(close_1d) >= atr_period:
        tr = np.maximum(np.maximum(high_1d[1:] - low_1d[1:], 
                                   np.abs(high_1d[1:] - close_1d[:-1])), 
                        np.abs(low_1d[1:] - close_1d[:-1]))
        atr[atr_period - 1] = np.mean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
        # Prepend NaN for index 0
        atr = np.concatenate([[np.nan], atr])
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50
    ema_period = 50
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= ema_period:
        ema_12h[ema_period - 1] = np.mean(close_12h[:ema_period])
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * (2 / (ema_period + 1)) + 
                         ema_12h[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Align indicators to 12h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Bollinger Bands width (20, 2) on 12h
    bb_period = 20
    bb_std = 2
    bb_ma = np.full(len(close_12h), np.nan)
    bb_stddev = np.full(len(close_12h), np.nan)
    for i in range(bb_period, len(close_12h)):
        bb_ma[i] = np.mean(close_12h[i-bb_period:i])
        bb_stddev[i] = np.std(close_12h[i-bb_period:i])
    bb_upper = bb_ma + bb_stddev * bb_std
    bb_lower = bb_ma - bb_stddev * bb_std
    bb_width = bb_upper - bb_lower
    bb_width_aligned = align_htf_to_ltf(prices, df_12h, bb_width)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need ATR, EMA, volume MA, BB width
    start_idx = max(atr_period, ema_period, vol_period, bb_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_aligned[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(bb_width_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        atr_val = atr_aligned[i]
        
        # Volatility regime: BB width > 20-day average of BB width
        bb_width_ma = np.mean(bb_width_aligned[max(0, i-20):i+1]) if i >= 20 else np.mean(bb_width_aligned[:i+1])
        high_vol = bb_width_aligned[i] > bb_width_ma
        
        if position == 0:
            # Long: Price > 12h EMA50 + volume spike + high volatility
            if (price > ema_12h_aligned[i] and 
                vol_ratio > 1.5 and 
                high_vol):
                signals[i] = size
                position = 1
            # Short: Price < 12h EMA50 + volume spike + high volatility
            elif (price < ema_12h_aligned[i] and 
                  vol_ratio > 1.5 and 
                  high_vol):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price < 12h EMA50 OR volatility drops OR ATR-based stop
            if (price < ema_12h_aligned[i] or 
                not high_vol or
                price < close[i-1] - 1.5 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price > 12h EMA50 OR volatility drops OR ATR-based stop
            if (price > ema_12h_aligned[i] or 
                not high_vol or
                price > close[i-1] + 1.5 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_EMA50_Volume_Volatility_ATRStop"
timeframe = "12h"
leverage = 1.0