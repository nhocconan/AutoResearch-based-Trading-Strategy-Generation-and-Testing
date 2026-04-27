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
    
    # Get 1-day and 1-week data for calculations (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1-week Exponential Moving Average (34-period) for long-term trend
    close_1w = df_1w['close'].values
    ema_34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        multiplier = 2 / (34 + 1)
        ema_34_1w[33] = np.mean(close_1w[:34])
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = (close_1w[i] * multiplier) + (ema_34_1w[i-1] * (1 - multiplier))
    
    # Calculate 1-day Donchian Channel (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donch_high_1d = np.full(len(high_1d), np.nan)
    donch_low_1d = np.full(len(low_1d), np.nan)
    
    if len(high_1d) >= 20:
        for i in range(19, len(high_1d)):
            donch_high_1d[i] = np.max(high_1d[i-19:i+1])
            donch_low_1d[i] = np.min(low_1d[i-19:i+1])
    
    # Align 1w EMA and 1d Donchian to 6h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # Calculate 6-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 6
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(34, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(donch_high_1d_aligned[i]) or 
            np.isnan(donch_low_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Price above weekly EMA34 and breaks above daily Donchian high with volume
            if price > ema_34_1w_aligned[i] and price > donch_high_1d_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: Price below weekly EMA34 and breaks below daily Donchian low with volume
            elif price < ema_34_1w_aligned[i] and price < donch_low_1d_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price crosses below weekly EMA34 or breaks below daily Donchian low
            if price < ema_34_1w_aligned[i] or price < donch_low_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price crosses above weekly EMA34 or breaks above daily Donchian high
            if price > ema_34_1w_aligned[i] or price > donch_high_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WEMA34_Donchian20_Volume"
timeframe = "6h"
leverage = 1.0