#!/usr/bin/env python3
"""
12h Donchian Breakout + Volume Confirmation + ATR Stoploss
Hypothesis: Donchian(20) breakouts capture strong trends with volume confirmation.
Works in bull (buy breakouts above upper band) and bear (sell breakdowns below lower band).
Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14382_12h_donchian20_1d_vol_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian and volume filters (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Donchian channels (20-period)
    donch_len = 20
    upper = pd.Series(high_1d).rolling(window=donch_len, min_periods=donch_len).max().values
    lower = pd.Series(low_1d).rolling(window=donch_len, min_periods=donch_len).min().values
    
    # Volume filter: above average volume
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = donch_len + 14  # Donchian + ATR
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR stoploss
            if (close[i] < lower_aligned[i] or
                close[i] <= entry_price - 2.5 * atr_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR stoploss
            if (close[i] > upper_aligned[i] or
                close[i] >= entry_price + 2.5 * atr_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume confirmation
            long_breakout = close[i] > upper_aligned[i]
            short_breakout = close[i] < lower_aligned[i]
            vol_ok = volume[i] > vol_ma_aligned[i]  # Volume above average
            
            if long_breakout and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals