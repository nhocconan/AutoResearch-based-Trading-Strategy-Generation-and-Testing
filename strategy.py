#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Choppiness Index + Donchian(20) breakout with volume confirmation
# Choppiness Index identifies trending vs ranging markets (CHOP > 61.8 = range, < 38.2 = trend).
# In trending markets, Donchian breakouts capture momentum; in ranging markets, fade at Donchian bands.
# Volume filter ensures breakouts have institutional participation.
# Works in bull/bear by adapting to market regime via Choppiness Index.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate Choppiness Index (14-period) on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr_1d = np.full(len(df_1d), np.nan)
    for i in range(1, len(df_1d)):
        hl = high_1d[i] - low_1d[i]
        hc = np.abs(high_1d[i] - close_1d[i-1])
        lc = np.abs(low_1d[i] - close_1d[i-1])
        tr_1d[i] = np.max([hl, hc, lc])
    
    # ATR(14) for 1d
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr_1d[i-14:i+1])
    
    # Choppiness Index: 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        sum_atr = np.sum(atr_1d[i-14:i+1])
        max_high = np.max(high_1d[i-14:i+1])
        min_low = np.min(low_1d[i-14:i+1])
        if max_high > min_low:
            chop_1d[i] = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14)
    
    # Align Choppiness Index to 6h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume filter: volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), Chop (14), Vol MA (20)
    start_idx = max(19, 14, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        chop = chop_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Regime detection
        is_trending = chop < 38.2
        is_ranging = chop > 61.8
        
        if position == 0:
            if is_trending and vol_filter:
                # In trending market: Donchian breakout
                if price > donchian_high[i]:
                    signals[i] = size
                    position = 1
                elif price < donchian_low[i]:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_ranging and vol_filter:
                # In ranging market: fade at Donchian bands (mean reversion)
                if price >= donchian_high[i]:
                    signals[i] = -size
                    position = -1
                elif price <= donchian_low[i]:
                    signals[i] = size
                    position = 1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to midpoint or Donchian low
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if price <= midpoint or price <= donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to midpoint or Donchian high
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if price >= midpoint or price >= donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Choppiness_Donchian_Breakout_Volume"
timeframe = "6h"
leverage = 1.0