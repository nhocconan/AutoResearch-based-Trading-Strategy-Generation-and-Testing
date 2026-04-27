#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index + Donchian breakout with volume confirmation
# Choppiness Index > 61.8 indicates ranging market (mean reversion opportunity)
# Donchian breakout in ranging conditions captures false breakout reversals
# Works in both bull and bear by using volatility regime rather than trend
# Target: 80-120 total trades over 4 years (~20-30/year) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR(14) - Average True Range
    atr_14 = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        atr_14[i] = np.mean(tr[i-13:i+1])
    
    # Choppiness Index: 100 * log(sum(ATR)/log(n)) / log(n)
    chop = np.full(len(df_1d), np.nan)
    for i in range(13, len(df_1d)):
        atr_sum = np.sum(tr[i-13:i+1])
        max_h = np.max(high_1d[i-13:i+1])
        min_l = np.min(low_1d[i-13:i+1])
        if max_h > min_l and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / (max_h - min_l)) / np.log10(14)
        else:
            chop[i] = 50.0  # Neutral when undefined
    
    # Align Choppiness Index to 4h timeframe (wait for 1d close)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian Channel (20-period) on 4h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume filter: volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d data (14), Donchian (20), volume MA (20)
    start_idx = max(14, 19, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Range filter: Choppiness > 61.8 indicates ranging market
        is_ranging = chop_aligned[i] > 61.8
        
        # Volume filter: above average volume
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: breakdown below Donchian low in ranging market (expect reversion up)
            if price < donchian_low[i] and is_ranging and vol_filter:
                signals[i] = size
                position = 1
            # Short: breakout above Donchian high in ranging market (expect reversion down)
            elif price > donchian_high[i] and is_ranging and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian mean or exits ranging condition
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if price >= donchian_mid or chop_aligned[i] <= 50:  # Exit ranging or reach midpoint
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to Donchian mean or exits ranging condition
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            if price <= donchian_mid or chop_aligned[i] <= 50:  # Exit ranging or reach midpoint
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Choppiness_Donchian_MeanReversion_Volume"
timeframe = "4h"
leverage = 1.0