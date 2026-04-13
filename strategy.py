#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d VWAP trend and volume confirmation.
# Long: price breaks above Donchian(20) high + price > 1d VWAP + volume > 1.5x avg volume
# Short: price breaks below Donchian(20) low + price < 1d VWAP + volume > 1.5x avg volume
# VWAP from 1d data acts as dynamic trend filter to avoid counter-trend breakouts
# Volume confirmation reduces false breakouts
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Works in both bull and bear markets by using 1d VWAP as trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate VWAP for each 1d bar (cumulative since start of day)
    vwap_1d = np.full(len(close_1d), np.nan)
    cum_vol = 0.0
    cum_price_vol = 0.0
    for i in range(len(close_1d)):
        typical_price = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
        cum_price_vol += typical_price * volume_1d[i]
        cum_vol += volume_1d[i]
        if cum_vol > 0:
            vwap_1d[i] = cum_price_vol / cum_vol
    
    # Align 1d VWAP to 12h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Donchian(20) on 12h timeframe
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
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
            np.isnan(vwap_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        vwap = vwap_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: break above Donchian high + above VWAP + volume confirmation
            if (price > donch_high[i] and 
                price > vwap and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: break below Donchian low + below VWAP + volume confirmation
            elif (price < donch_low[i] and 
                  price < vwap and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or below VWAP
            if (price < donch_low[i] or
                price < vwap):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high or above VWAP
            if (price > donch_high[i] or
                price > vwap):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian_VWAP_Volume"
timeframe = "12h"
leverage = 1.0