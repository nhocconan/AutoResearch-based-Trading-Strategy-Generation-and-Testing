#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and Regime Filter
Hypothesis: Breakouts from Donchian channels (20) on 4h timeframe, confirmed by volume surge and filtered by Choppy market regime, capture institutional moves while avoiding false breakouts in sideways markets. Works in both bull (long breakouts) and bear (short breakdowns) by using symmetric entry conditions.
Target: 100-200 total trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_volume_chop_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Chop regime filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Choppy market regime (14-period) on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over last 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation: log(sum(TR)/range) / log(14)
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    range_1d = hh_1d - ll_1d
    chop = 100 * np.log10(sum_tr / (range_1d + 1e-10)) / np.log10(14)
    chop_filter = chop > 61.8  # Choppy market threshold
    
    # 4h data for Donchian breakout
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Donchian channels (20-period) on 4h
    dc_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    dc_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume_4h > (1.5 * vol_ma)
    
    # ATR for stoploss calculation
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF Chop filter to 4h timeframe
    chop_filter_4h = align_htf_to_ltf(prices, df_1d, chop_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For Donchian channels
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr_4h[i]) or
            np.isnan(chop_filter_4h[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Avoid choppy markets - only trade in trending conditions
        if chop_filter_4h[i]:
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: breakdown below Donchian low OR stoploss
            if (close_4h[i] <= dc_low[i] or
                close_4h[i] <= entry_price - 2.5 * atr_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: breakout above Donchian high OR stoploss
            if (close_4h[i] >= dc_high[i] or
                close_4h[i] >= entry_price + 2.5 * atr_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume confirmation
            long_breakout = close_4h[i] > dc_high[i]
            short_breakout = close_4h[i] < dc_low[i]
            
            if long_breakout and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close_4h[i]
            elif short_breakout and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close_4h[i]
            else:
                signals[i] = 0.0
    
    return signals