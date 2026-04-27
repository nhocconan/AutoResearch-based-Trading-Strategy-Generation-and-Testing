#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d Volume Spike and ATR Stoploss.
Long when price breaks above Donchian(20) high + volume spike.
Short when price breaks below Donchian(20) low + volume spike.
Exit when price returns to Donchian midline or ATR trailing stop.
Designed to generate 25-40 trades/year per symbol with strong edge in both bull and bear.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d average volume (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = np.empty_like(vol_1d, dtype=np.float64)
    vol_ma_20_1d.fill(np.nan)
    for i in range(19, len(vol_1d)):
        vol_ma_20_1d[i] = np.mean(vol_1d[i-19:i+1])
    
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.empty_like(high, dtype=np.float64)
    donchian_low = np.empty_like(low, dtype=np.float64)
    donchian_high.fill(np.nan)
    donchian_low.fill(np.nan)
    
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.empty_like(tr, dtype=np.float64)
    atr.fill(np.nan)
    for i in range(13, n):
        if i == 13:
            atr[i] = np.mean(tr[0:14])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Track highest high since entry for trailing stop
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: need Donchian (20), ATR (14), volume MA (20)
    start_idx = max(19, 13, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            # Reset tracking when flat
            if position == 0:
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        dc_high = donchian_high[i]
        dc_low = donchian_low[i]
        dc_mid = donchian_mid[i]
        atr_now = atr[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Volume filter: volume > 2.0x 1d average
        vol_filter = vol_now > 2.0 * vol_ma
        
        if position == 0:
            # Track new high/low for potential breakout
            highest_since_entry = price_now
            lowest_since_entry = price_now
            
            # Long: price breaks above Donchian high + volume spike
            if price_now > dc_high and vol_filter:
                signals[i] = size
                position = 1
                highest_since_entry = price_now
            # Short: price breaks below Donchian low + volume spike
            elif price_now < dc_low and vol_filter:
                signals[i] = -size
                position = -1
                lowest_since_entry = price_now
            else:
                signals[i] = 0.0
        elif position == 1:
            # Update highest since entry
            if price_now > highest_since_entry:
                highest_since_entry = price_now
            
            # Exit conditions:
            # 1. Price returns to Donchian midline
            # 2. ATR trailing stop (3 * ATR from high)
            if price_now < dc_mid or price_now < highest_since_entry - 3.0 * atr_now:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Update lowest since entry
            if price_now < lowest_since_entry:
                lowest_since_entry = price_now
            
            # Exit conditions:
            # 1. Price returns to Donchian midline
            # 2. ATR trailing stop (3 * ATR from low)
            if price_now > dc_mid or price_now > lowest_since_entry + 3.0 * atr_now:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0