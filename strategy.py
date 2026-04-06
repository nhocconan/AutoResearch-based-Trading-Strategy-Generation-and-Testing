#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + Volume Spike + Choppiness Regime Filter
Hypothesis: Donchian breakouts capture strong moves; volume spike confirms institutional participation;
choppiness regime filter avoids whipsaws in ranging markets. Works in bull (buy breakouts) and bear
(sell breakdowns). Target: 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14408_12h_donchian20_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1w data for choppiness regime (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Choppiness Index (14-period)
    chop_period = 14
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w[0] = tr1[0]
    atr_1w = pd.Series(tr_1w).rolling(window=chop_period, min_periods=chop_period).sum().values
    max_h = pd.Series(high_1w).rolling(window=chop_period, min_periods=chop_period).max().values
    min_l = pd.Series(low_1w).rolling(window=chop_period, min_periods=chop_period).min().values
    range_max_min = max_h - min_l
    chop = 100 * np.log10(atr_1w / range_max_min) / np.log10(chop_period)
    chop[range_max_min == 0] = 50  # avoid division by zero
    
    # Align choppiness to 12h
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Load 1d data for Donchian channels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian channels (20-period)
    donch_period = 20
    upper = pd.Series(high_1d).rolling(window=donch_period, min_periods=donch_period).max().values
    lower = pd.Series(low_1d).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Align Donchian to 12h
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(14, 20, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(chop_aligned[i]) or np.isnan(upper_aligned[i]) or
            np.isnan(lower_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Regime filter: chop > 61.8 = ranging (mean revert), chop < 38.2 = trending (trend follow)
        chop_val = chop_aligned[i]
        is_ranging = chop_val > 61.8
        is_trending = chop_val < 38.2
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR chop becomes ranging OR stoploss
            if (close[i] < lower_aligned[i] or is_ranging or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR chop becomes ranging OR stoploss
            if (close[i] > upper_aligned[i] or is_ranging or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume spike + trend regime
            long_setup = (close[i] > upper_aligned[i] and vol_spike[i] and is_trending)
            short_setup = (close[i] < lower_aligned[i] and vol_spike[i] and is_trending)
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals