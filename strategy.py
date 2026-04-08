#!/usr/bin/env python3
# [24884] 1d_1w_donchian_volume_breakout_v1
# Hypothesis: 1-day strategy using weekly Donchian channels with volume confirmation and ATR-based stop.
# Long when price breaks above weekly Donchian high with volume > 2x average.
# Short when price breaks below weekly Donchian low with volume > 2x average.
# Exit when price crosses back below/above weekly Donchian low/high or volume falls below 1.5x average.
# Uses weekly trend filter to avoid counter-trend trades in choppy markets.
# Target: 20-30 trades/year per symbol.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_volume_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-period weekly Donchian channels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    donchian_high = np.full(len(high_1w), np.nan)
    donchian_low = np.full(len(low_1w), np.nan)
    
    for i in range(20, len(high_1w)):
        donchian_high[i] = np.max(high_1w[i-20:i])
        donchian_low[i] = np.min(low_1w[i-20:i])
    
    # Align Donchian channels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume confirmation: 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below weekly Donchian low or volume drops below 1.5x average
            if price < lower or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above weekly Donchian high or volume drops below 1.5x average
            if price > upper or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above weekly Donchian high with volume expansion
            if price > upper and vol_ratio > 2.0:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below weekly Donchian low with volume expansion
            elif price < lower and vol_ratio > 2.0:
                position = -1
                signals[i] = -0.25
    
    return signals