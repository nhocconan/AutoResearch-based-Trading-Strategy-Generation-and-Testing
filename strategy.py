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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly high/low for Donchian channel
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian(20) channels
    donch_high = np.full(len(close_1w), np.nan)
    donch_low = np.full(len(close_1w), np.nan)
    for i in range(20, len(close_1w)):
        donch_high[i] = np.max(high_1w[i-20:i])
        donch_low[i] = np.min(low_1w[i-20:i])
    
    # Align weekly Donchian to daily
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # Calculate 20-period volume average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(20, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high with volume confirmation
            if price > donch_high_aligned[i] and vol_ratio > 2.0:
                signals[i] = size
                position = 1
            # Short: Price breaks below weekly Donchian low with volume confirmation
            elif price < donch_low_aligned[i] and vol_ratio > 2.0:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below weekly Donchian low
            if price < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above weekly Donchian high
            if price > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyDonchian20_Volume_Breakout"
timeframe = "1d"
leverage = 1.0