#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with daily pivot direction and volume confirmation
# Works in bull/bear by using daily pivot levels as trend filters (price above daily pivot = bullish bias, below = bearish bias)
# Donchian breakouts capture momentum, pivot direction avoids counter-trend trades, volume confirms legitimacy
# Target: 15-35 trades/year (~60-140 total over 4 years) to minimize fee drag

name = "6h_1d_donchian_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot point (using prior day's OHLC)
    pivot = np.full(len(df_1d), np.nan)
    for i in range(1, len(df_1d)):
        ph = float(df_1d['high'].iloc[i-1])
        pl = float(df_1d['low'].iloc[i-1])
        pc = float(df_1d['close'].iloc[i-1])
        pivot[i] = (ph + pl + pc) / 3.0
    
    # Align daily pivot to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Donchian channel (20-period) on 6h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: 4-period average (24h)
    vol_ma_4 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 4:
            vol_sum -= volume[i-4]
        if i >= 3:
            vol_ma_4[i] = vol_sum / 4
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(pivot_6h[i]) or 
            np.isnan(vol_ma_4[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR pivot bias turns bearish
            if (close[i] <= donchian_low[i]) or (close[i] < pivot_6h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR pivot bias turns bullish
            if (close[i] >= donchian_high[i]) or (close[i] > pivot_6h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above Donchian high with volume confirmation AND price above daily pivot (bullish bias)
            vol_ratio = volume[i] / vol_ma_4[i] if vol_ma_4[i] > 0 else 0
            if (close[i] > donchian_high[i] and 
                vol_ratio > 1.5 and 
                close[i] > pivot_6h[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below Donchian low with volume confirmation AND price below daily pivot (bearish bias)
            elif (close[i] < donchian_low[i] and 
                  vol_ratio > 1.5 and 
                  close[i] < pivot_6h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals