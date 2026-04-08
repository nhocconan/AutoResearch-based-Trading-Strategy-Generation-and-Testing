#!/usr/bin/env python3
# 4h_1d_donchian_breakout_volume_regime_v1
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d chop regime filter.
# Long when price breaks above upper band + volume > 1.5x average + chop > 61.8 (range).
# Short when price breaks below lower band + volume > 1.5x average + chop > 61.8.
# Uses 1d chop filter to avoid false breakouts in strong trends, favoring mean reversion in ranges.
# Designed for 20-50 trades/year on 4h to avoid fee drag. Works in bull/bear via regime filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_20 = np.full(n, np.nan)
    low_20 = np.full(n, np.nan)
    for i in range(20, n):
        high_20[i] = np.max(high[i-20:i+1])
        low_20[i] = np.min(low[i-20:i+1])
    
    # Average volume (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i+1])
    
    # Get 1d data for chop filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for chop calculation
    tr_1d = np.full(len(df_1d), np.nan)
    for i in range(1, len(df_1d)):
        tr_1d[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    # ATR(14) for chop denominator
    atr_14 = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_14[i] = np.mean(tr_1d[i-14:i+1])
    
    # Sum of true range over 14 periods
    tr_sum_14 = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        tr_sum_14[i] = np.sum(tr_1d[i-14:i+1])
    
    # Chop formula: 100 * log10(tr_sum_14 / (atr_14 * 14)) / log10(14)
    chop = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        if atr_14[i] > 0 and tr_sum_14[i] > 0:
            chop[i] = 100 * np.log10(tr_sum_14[i] / (atr_14[i] * 14)) / np.log10(14)
    
    # Align chop to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(20, 14)  # Ensure Donchian and chop are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band
            if close[i] < low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band
            if close[i] > high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average
            vol_confirm = volume[i] > 1.5 * vol_ma[i]
            
            # Chop filter: chop > 61.8 indicates ranging market (favor mean reversion)
            chop_filter = chop_aligned[i] > 61.8
            
            # Long entry: price breaks above upper band + volume + chop filter
            if (close[i] > high_20[i] and vol_confirm and chop_filter):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower band + volume + chop filter
            elif (close[i] < low_20[i] and vol_confirm and chop_filter):
                position = -1
                signals[i] = -0.25
    
    return signals