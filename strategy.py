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
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 20-day Donchian channels (upper/lower)
    highest_20 = np.full(len(df_1d), np.nan)
    lowest_20 = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i >= 19:  # 20 periods needed
            highest_20[i] = np.max(df_1d['high'].iloc[i-19:i+1])
            lowest_20[i] = np.min(df_1d['low'].iloc[i-19:i+1])
    
    # Calculate 20-day ATR for volatility filter
    tr = np.zeros(len(df_1d))
    tr[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]
    for i in range(1, len(df_1d)):
        hl = df_1d['high'].iloc[i] - df_1d['low'].iloc[i]
        hc = abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1])
        lc = abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr_20 = np.full(len(tr), np.nan)
    for i in range(19, len(tr)):
        if i == 19:
            atr_20[i] = np.mean(tr[:20])
        else:
            atr_20[i] = (atr_20[i-1] * 19 + tr[i]) / 20
    
    # Align indicators to 12h timeframe
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    
    # Calculate volume ratio (current volume / 20-day average volume)
    vol_ma_20 = np.full(len(df_1d), np.nan)
    vol_series = df_1d['volume'].values
    for i in range(len(vol_ma_20)):
        if i >= 19:
            vol_ma_20[i] = np.mean(vol_series[i-19:i+1])
    
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup period
    start_idx = 19
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(atr_20_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = highest_20_aligned[i]
        lower = lowest_20_aligned[i]
        atr = atr_20_aligned[i]
        vol_ratio = volume[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 1.0
        
        if position == 0:
            # Long: Break above upper Donchian with volume confirmation
            if price > upper and vol_ratio > 1.5:
                signals[i] = size
                position = 1
            # Short: Break below lower Donchian with volume confirmation
            elif price < lower and vol_ratio > 1.5:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price crosses below midpoint or ATR-based stop
            midpoint = (upper + lower) / 2
            if price < midpoint or price < close[i-1] - 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Price crosses above midpoint or ATR-based stop
            midpoint = (upper + lower) / 2
            if price > midpoint or price > close[i-1] + 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian_Breakout_Volume_ATR_Stop"
timeframe = "12h"
leverage = 1.0