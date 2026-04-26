#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_ChopFilter_v1
Hypothesis: Donchian(20) breakout with volume spike confirmation and choppiness regime filter on 4h timeframe. 
In choppy markets (CHOP > 61.8), we fade breakouts (mean reversion); in trending markets (CHOP < 38.2), we follow breakouts.
This adapts to both bull and bear regimes while minimizing false breakouts. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR(14) for choppiness index
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14)
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR14, n) / (n * ATR1)) / log10(n)
    # where n=14 period
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(atr_sum / (14 * (highest_high - lowest_low))) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)  # neutral when undefined
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian(20) channels on 4h
    lookback = 20
    highest_high_4h = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low_4h = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike detection: volume > 2.0x 20-period EMA
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(lookback, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(highest_high_4h[i]) or
            np.isnan(lowest_low_4h[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filter based on 1d Choppiness Index
        chop_value = chop_aligned[i]
        is_choppy = chop_value > 61.8  # range market -> mean reversion
        is_trending = chop_value < 38.2  # trending market -> follow breakout
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high_4h[i-1]  # break above previous high
        breakout_down = close[i] < lowest_low_4h[i-1]  # break below previous low
        
        # Long logic
        if is_trending and breakout_up and volume_spike[i]:
            # Trending market: follow breakout
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        elif is_choppy and breakout_down and volume_spike[i]:
            # Choppy market: fade breakout (sell the breakdown)
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Short logic
        elif is_trending and breakout_down and volume_spike[i]:
            # Trending market: follow breakout down
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        elif is_choppy and breakout_up and volume_spike[i]:
            # Choppy market: fade breakout (buy the breakout failure)
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Exit conditions: opposite Donchian breakout or volatility contraction
        elif position == 1 and (breakout_down or chop_value > 70):  # exit on breakdown or extreme chop
            signals[i] = 0.0
            position = 0
        elif position == -1 and (breakout_up or chop_value > 70):  # exit on breakout or extreme chop
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0