#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + Volume Spike + Choppiness Regime Filter
Hypothesis: Donchian breakouts capture strong momentum moves. Volume confirms conviction.
Choppiness filter avoids whipsaws in sideways markets. Works in both bull (breakouts up)
and bear (breakouts down) by trading the breakout direction. Target: 80-120 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14462_12h_donchian20_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Donchian channels and choppiness (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels (20-period)
    donchian_period = 20
    upper_channel = pd.Series(high_1d).rolling(window=donchian_period, min_periods=donchian_period).max()
    lower_channel = pd.Series(low_1d).rolling(window=donchian_period, min_periods=donchian_period).min()
    
    # Choppiness Index (14-period) - range vs trend filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    
    # Avoid division by zero
    range_14 = highest_high_14 - lowest_low_14
    chop = np.where(range_14 > 0, 100 * np.log10(sum_tr_14 / range_14) / np.log10(14), 50)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: spike above average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)  # Require 150% of average volume
    
    # ATR for stoploss (12h)
    tr1_12h = high - low
    tr2_12h = np.abs(high - np.roll(close, 1))
    tr3_12h = np.abs(low - np.roll(close, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_channel.values)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_channel.values)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = donchian_period + 14  # Donchian(20) + Chop(14)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_12h[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Choppiness filter: avoid choppy markets (Chop > 61.8 = range)
        # Only trade when Chop <= 61.8 (trending)
        if chop_aligned[i] > 61.8:
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below lower channel OR stoploss
            if (close[i] < lower_aligned[i] or
                close[i] <= entry_price - 2.5 * atr_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper channel OR stoploss
            if (close[i] > upper_aligned[i] or
                close[i] >= entry_price + 2.5 * atr_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume spike
            long_breakout = close[i] > upper_aligned[i]
            short_breakout = close[i] < lower_aligned[i]
            
            if long_breakout and vol_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and vol_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals