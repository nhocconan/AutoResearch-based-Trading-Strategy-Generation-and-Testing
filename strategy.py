#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h with 4h/1d trend filter and volume confirmation
# Uses 4h Donchian breakouts for signal direction, 1d EMA200 for long-term trend filter, and volume spike for confirmation
# Trades only during active session (08-20 UTC) to reduce noise
# Target: 15-37 trades/year (60-150 over 4 years) to minimize fee drag
# Works in both bull and bear markets by requiring alignment with long-term trend and volume confirmation

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (called ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for trend filter (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 4-period Donchian channels (20-period lookback)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_high = np.full(len(df_4h), np.nan)
    donchian_low = np.full(len(df_4h), np.nan)
    
    for i in range(19, len(df_4h)):
        donchian_high[i] = np.max(high_4h[i-19:i+1])
        donchian_low[i] = np.min(low_4h[i-19:i+1])
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        multiplier = 2 / (200 + 1)
        ema_200_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_200_1d[i] = (close_1d[i] * multiplier) + (ema_200_1d[i-1] * (1 - multiplier))
    
    # Calculate 4-period average volume for spike detection
    vol_ma_4h = np.full(len(df_4h), np.nan)
    vol_period = 4
    for i in range(vol_period, len(df_4h)):
        vol_ma_4h[i] = np.mean(df_4h['volume'].values[i-vol_period:i])
    
    # Align all indicators to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0
    size = 0.20
    
    # Warmup period
    start_idx = max(20, 200) + 5
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
            
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_4h_aligned[i] if vol_ma_4h_aligned[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume and 1d uptrend
            if price > donchian_high_aligned[i] and price > ema_200_1d_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: Price breaks below Donchian low with volume and 1d downtrend
            elif price < donchian_low_aligned[i] and price < ema_200_1d_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price breaks below Donchian low or trend reverses
            if price < donchian_low_aligned[i] or price < ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price breaks above Donchian high or trend reverses
            if price > donchian_high_aligned[i] or price > ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Donchian_20_1dEMA200_Volume_Session"
timeframe = "1h"
leverage = 1.0