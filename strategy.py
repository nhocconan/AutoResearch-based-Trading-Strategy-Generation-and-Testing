#!/usr/bin/env python3
# [24936] 12h_1d_donchian_volume_regime_v1
# Hypothesis: 12-hour Donchian(20) breakout with 1-day volume confirmation and chop regime filter.
# Long when price breaks above 20-period high with volume > 2.0x average and chop > 61.8 (range).
# Short when price breaks below 20-period low with volume > 2.0x average and chop > 61.8 (range).
# Exit when price returns to opposite Donchian level or chop < 38.2 (trend).
# Uses chop regime to avoid false breakouts in strong trends, effective in choppy markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_donchian_volume_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate chop index (14-period) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.zeros(len(close_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(close_1d)):
        tr[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    
    # ATR (14-period)
    atr = np.zeros(len(close_1d))
    atr[13] = np.mean(tr[0:14])
    for i in range(14, len(close_1d)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Chop = 100 * log10(sum(ATR,14) / (max(high,14) - min(low,14))) / log10(14)
    chop = np.full(len(close_1d), np.nan)
    for i in range(13, len(close_1d)):
        atr_sum = np.sum(atr[i-13:i+1])
        max_high = np.max(high_1d[i-13:i+1])
        min_low = np.min(low_1d[i-13:i+1])
        if max_high > min_low:
            chop[i] = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
        else:
            chop[i] = 50.0  # neutral if no range
    
    # Calculate Donchian channels (20-period) on 12h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align chop to 12-hour timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        chop_val = chop_aligned[i]
        
        if position == 1:  # Long
            # Exit: price returns to Donchian low or chop < 38.2 (trending)
            if price <= donchian_low[i] or chop_val < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to Donchian high or chop < 38.2 (trending)
            if price >= donchian_high[i] or chop_val < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume expansion and chop > 61.8 (range)
            if price > donchian_high[i] and vol_ratio > 2.0 and chop_val > 61.8:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume expansion and chop > 61.8 (range)
            elif price < donchian_low[i] and vol_ratio > 2.0 and chop_val > 61.8:
                position = -1
                signals[i] = -0.25
    
    return signals