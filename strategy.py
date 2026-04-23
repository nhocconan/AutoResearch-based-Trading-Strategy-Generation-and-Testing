#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume spike and choppiness regime filter.
Long when price breaks above Donchian upper(20) AND volume > 2.0x 20-period MA AND chop > 61.8 (range).
Short when price breaks below Donchian lower(20) AND volume > 2.0x 20-period MA AND chop > 61.8 (range).
Exit when price touches opposite Donchian level.
Uses volume spike for momentum confirmation and chop filter to avoid false breakouts in strong trends.
Targets 75-200 trades over 4 years (19-50/year) for 4h timeframe.
Donchian provides structure, volume confirms momentum, chop filter ensures mean-reversion context.
Works in bull/bear by fading breakouts in ranging markets (chop > 61.8).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        donchian_upper[i] = np.max(high[i-lookback+1:i+1])
        donchian_lower[i] = np.min(low[i-lookback+1:i+1])
    
    # Calculate 1d ATR(14) for choppiness indicator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with 1d indices
    
    # ATR(14)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR14) / (max(high) - min(low)) over period) / log10(period)
    chop_period = 14
    atr_sum = pd.Series(atr_1d).rolling(window=chop_period, min_periods=chop_period).sum().values
    max_high = pd.Series(high_1d).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low_1d).rolling(window=chop_period, min_periods=chop_period).min().values
    chop_denom = max_high - min_low
    chop_raw = 100 * np.log10(atr_sum / chop_denom) / np.log10(chop_period)
    chop_1d = np.where(chop_denom > 0, chop_raw, np.nan)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 4h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback - 1, 14 + 14 - 1, 20)  # Donchian, chop (needs 2*14-1), volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        chop_val = chop_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Volume filter: 4h volume > 2.0x 20-period MA
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        # Chop filter: > 61.8 indicates ranging market (mean reversion context)
        chop_filter = chop_val > 61.8
        
        if position == 0:
            # Long: Break above Donchian upper AND volume filter AND chop filter (range)
            if price > upper and vol_filter and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower AND volume filter AND chop filter (range)
            elif price < lower and vol_filter and chop_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price touches opposite Donchian level
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches lower Donchian
                if price < lower:
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches upper Donchian
                if price > upper:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0