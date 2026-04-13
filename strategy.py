#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d Donchian(10) trend filter and volume confirmation.
# Long: price breaks above 12h Donchian(20) high + price above 1d Donchian(10) high + volume > 1.5x avg volume
# Short: price breaks below 12h Donchian(20) low + price below 1d Donchian(10) low + volume > 1.5x avg volume
# Volume confirmation reduces false breakouts
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Works in both bull and bear markets by using higher timeframe Donchian as trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 12h Donchian(20)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # 1d Donchian(10) for trend filter
    donch_high_1d = np.full(len(high_1d), np.nan)
    donch_low_1d = np.full(len(low_1d), np.nan)
    for i in range(10, len(high_1d)):
        donch_high_1d[i] = np.max(high_1d[i-10:i])
        donch_low_1d[i] = np.min(low_1d[i-10:i])
    
    # Align 1d Donchian to 12h timeframe
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # Average volume (20-period = 20*12h = 10 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(donch_high_1d_aligned[i]) or np.isnan(donch_low_1d_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: break above 12h Donchian high + above 1d Donchian high + volume confirmation
            if (price > donch_high[i] and 
                price > donch_high_1d_aligned[i] and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: break below 12h Donchian low + below 1d Donchian low + volume confirmation
            elif (price < donch_low[i] and 
                  price < donch_low_1d_aligned[i] and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below 12h Donchian low or below 1d Donchian low
            if (price < donch_low[i] or
                price < donch_low_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above 12h Donchian high or above 1d Donchian high
            if (price > donch_high[i] or
                price > donch_high_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian_Trend_Volume"
timeframe = "12h"
leverage = 1.0