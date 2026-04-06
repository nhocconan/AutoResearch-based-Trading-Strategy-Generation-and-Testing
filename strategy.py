#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12820_4h_1d_donchian_vol"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-day high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Use pandas rolling for Donchian calculation
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    
    # Donchian high: highest high of last 20 days
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low of last 20 days
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe (shifted by 1 day for completed bars)
    donchian_high_4h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_4h = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(20, 20, 14)  # Donchian, volume MA, ATR periods
    
    for i in range(start, n):
        # Skip if Donchian levels not available
        if np.isnan(donchian_high_4h[i]) or np.isnan(donchian_low_4h[i]):
            if position != 0:
                signals[i] = 0.30 if position == 1 else -0.30
            else:
                signals[i] = 0.0
            continue
        
        # Check stop loss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.5) if not np.isnan(volume_ma[i]) else False
        
        # Breakout signals
        breakout_long = volume_ok and close[i] >= donchian_high_4h[i]
        breakout_short = volume_ok and close[i] <= donchian_low_4h[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif breakout_short:
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = 0.30
        elif position == -1:
            signals[i] = -0.30
    
    return signals