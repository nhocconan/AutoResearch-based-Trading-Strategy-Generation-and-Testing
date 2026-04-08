#!/usr/bin/env python3
# [24945] 12h_1d_donchian_volume_regime_v1
# Hypothesis: 12-hour Donchian(20) breakout with volume confirmation and choppiness regime filter.
# Long when price breaks above 20-period high with volume > 1.8x average and chop > 61.8 (ranging).
# Short when price breaks below 20-period low with volume > 1.8x average and chop > 61.8 (ranging).
# Exit when price reverts to 10-period moving average or volume drops below 1.3x average.
# Uses 1-day ATR for chop calculation to filter trending markets. Works in both bull/bear by focusing on mean reversion in ranging markets.

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
    
    # Calculate ATR(14) for chop
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_14 = np.full(len(close_1d), np.nan)
    for i in range(14, len(tr)):
        atr_14[i] = np.nanmean(tr[i-13:i+1])
    
    # Calculate Donchian(20) on 1-day for chop denominator
    donchian_high_1d = np.full(len(close_1d), np.nan)
    donchian_low_1d = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        donchian_high_1d[i] = np.max(high_1d[i-20:i])
        donchian_low_1d[i] = np.min(low_1d[i-20:i])
    
    # Chop = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    chop_1d = np.full(len(close_1d), np.nan)
    for i in range(33, len(close_1d)):  # Need 14 ATR + 20 Donchian
        sum_atr = np.nansum(atr_14[i-13:i+1])
        max_high = donchian_high_1d[i]
        min_low = donchian_low_1d[i]
        if max_high > min_low and not np.isnan(sum_atr):
            chop_1d[i] = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14)
        else:
            chop_1d[i] = np.nan
    
    # Calculate Donchian channels (20-period) for 12h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 10-period moving average for exit
    ma_10 = np.full(n, np.nan)
    for i in range(10, n):
        ma_10[i] = np.mean(close[i-10:i])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align chop to 12h timeframe
    chop_12h_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(35, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ma_10[i]) or np.isnan(vol_ma[i]) or np.isnan(chop_12h_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        chop = chop_12h_aligned[i]
        
        if position == 1:  # Long
            # Exit: price returns to 10-period MA or volume drops below 1.3x average
            if price <= ma_10[i] or vol_ratio < 1.3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to 10-period MA or volume drops below 1.3x average
            if price >= ma_10[i] or vol_ratio < 1.3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume expansion and chop > 61.8 (ranging)
            if price > donchian_high[i] and vol_ratio > 1.8 and chop > 61.8:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume expansion and chop > 61.8 (ranging)
            elif price < donchian_low[i] and vol_ratio > 1.8 and chop > 61.8:
                position = -1
                signals[i] = -0.25
    
    return signals