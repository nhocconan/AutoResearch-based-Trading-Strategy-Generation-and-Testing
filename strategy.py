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
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_period = 50
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period + 1)) + 
                        ema_1w[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Get daily data for ATR and Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period ATR for volatility filter
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                               np.abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = np.full(len(tr), np.nan)
    atr_period = 20
    for i in range(atr_period, len(tr)):
        atr_1d[i] = np.mean(tr[i-atr_period:i])
    
    # Calculate 20-period Donchian channels
    donch_high = np.full(len(high_1d), np.nan)
    donch_low = np.full(len(low_1d), np.nan)
    lookback = 20
    for i in range(lookback, len(high_1d)):
        donch_high[i] = np.max(high_1d[i-lookback:i])
        donch_low[i] = np.min(low_1d[i-lookback:i])
    
    # Align all indicators to daily timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need all indicators ready
    start_idx = max(ema_period, atr_period, lookback, vol_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        atr = atr_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + weekly uptrend + volume surge
            if (price > donch_high_aligned[i] and 
                price > ema_1w_aligned[i] and 
                vol_ratio > 1.5):
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low + weekly downtrend + volume surge
            elif (price < donch_low_aligned[i] and 
                  price < ema_1w_aligned[i] and 
                  vol_ratio > 1.5):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian low OR trend reversal
            if (price < donch_low_aligned[i] or 
                price < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above Donchian high OR trend reversal
            if (price > donch_high_aligned[i] or 
                price > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1D_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0