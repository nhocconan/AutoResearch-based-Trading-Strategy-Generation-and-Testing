#!/usr/bin/env python3
"""
4h_1d_1w_KAMA_Trend_with_Volume_and_Regime_Filter
Hypothesis: 4h timeframe using KAMA trend from 1d, volume confirmation, and weekly chop regime filter.
Designed to work in both bull and bear markets by only taking trend-aligned entries when market is not choppy.
KAMA adapts to market efficiency, reducing whipsaw in ranging markets. Volume confirms institutional participation.
Weekly chop filter avoids ranging markets where trend strategies fail. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_KAMA_Trend_with_Volume_and_Regime_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for KAMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (2, 10, 30) on daily close
    close_1d = df_1d['close'].values
    direction = np.abs(np.diff(close_1d, 10))  # net change over 10 periods
    volatility = np.sum(np.abs(np.diff(close_1d, 1)), axis=1)  # sum of absolute changes
    volatility = np.concatenate([np.full(10, np.nan), volatility])  # align with direction
    
    # Avoid division by zero
    er = np.where(volatility > 0, direction / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full_like(close_1d, np.nan)
    kama[29] = close_1d[29]  # seed
    for i in range(30, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_1d = kama
    
    # Align KAMA to 4h timeframe
    kama_4h = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Load 1w data ONCE before loop for chop regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate Choppy Index (14) on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])
    
    # Sum of true ranges over 14 periods
    atr14 = np.convolve(tr, np.ones(14), 'same') / 14
    atr14[:13] = np.nan  # pad beginning
    
    # Highest high and lowest low over 14 periods
    hh14 = np.convolve(high_1w, np.ones(14), 'same') / 14
    ll14 = np.convolve(low_1w, np.ones(14), 'same') / 14
    hh14[:13] = np.nan
    ll14[:13] = np.nan
    
    # Choppy Index
    chop = 100 * np.log10(atr14 / (hh14 - ll14)) / np.log10(14)
    chop = np.where((hh14 - ll14) > 0, chop, 50)  # avoid division by zero
    
    # Align chop to 4h timeframe
    chop_4h = align_htf_to_ltf(prices, df_1w, chop)
    
    # Volume average (20 period) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_4h[i]) or np.isnan(chop_4h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: avoid choppy markets (chop > 61.8)
        not_choppy = chop_4h[i] <= 61.8
        
        # Volume spike: current volume > 1.3x average
        volume_spike = volume[i] > vol_ma[i] * 1.3
        
        # Trend: price relative to KAMA
        above_kama = close[i] > kama_4h[i]
        below_kama = close[i] < kama_4h[i]
        
        # Entry conditions
        long_entry = above_kama and volume_spike and not_choppy
        short_entry = below_kama and volume_spike and not_choppy
        
        # Exit conditions: opposite KAMA touch or chop increases
        long_exit = below_kama or (chop_4h[i] > 61.8)
        short_exit = above_kama or (chop_4h[i] > 61.8)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals