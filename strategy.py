#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian(20) breakout + volume confirmation
# Long when price breaks above Donchian high (20) in trending regime (CHOP < 38.2) with volume spike
# Short when price breaks below Donchian low (20) in trending regime (CHOP < 38.2) with volume spike
# Uses 1d ATR for Choppiness Index to filter choppy markets and avoid false breakouts
# Volume spike confirms momentum. Designed for ~30-50 trades/year to minimize fee drag.
# Works in both bull and bear markets by only trading in clear trends.
name = "4h_Choppiness_Donchian20_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d True Range for Choppiness Index
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original array
    
    # ATR(14) for denominator
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Sum of TRUE RANGE over 14 periods for numerator
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: CHOP = 100 * log10(tr_sum_14 / (atr_14 * 14)) / log10(14)
    chop_raw = 100 * np.log10(tr_sum_14 / (atr_14 * 14)) / np.log10(14)
    
    # Align Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection (20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema_20 > 0, volume / vol_ema_20, 1.0) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(chop_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trending regime: Choppiness Index < 38.2
        trending = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long condition: break above Donchian high, trending regime, volume spike
            long_condition = (close[i] > highest_high[i]) and trending and vol_spike[i]
            # Short condition: break below Donchian low, trending regime, volume spike
            short_condition = (close[i] < lowest_low[i]) and trending and vol_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian low or regime turns choppy
            if (close[i] < lowest_low[i]) or (not trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian high or regime turns choppy
            if (close[i] > highest_high[i]) or (not trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals