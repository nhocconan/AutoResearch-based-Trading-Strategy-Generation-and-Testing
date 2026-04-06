#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume Confirmation and Chop Filter
Hypothesis: 12h Donchian(20) breakouts capture major trends on BTC/ETH/SOL. Volume confirms breakout strength, and chop filter avoids whipsaws in ranging markets. Works in bull (long breakouts) and bear (short breakdowns). Target: 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_volume_chop_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for chop filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Donchian channels (20-period) on 12h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Chop filter on 1d: Chop > 61.8 = ranging (avoid), Chop < 38.2 = trending (trade)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Chop = 100 * log15(sum(TR)/ (max(high)-min(low)))
    max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_tr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    denom_1d = np.maximum(max_high_1d - min_low_1d, 1e-10)
    chop = 100 * (np.log10(sum_tr_1d / denom_1d) / np.log10(14))
    
    # Chop thresholds: < 38.2 = trending, > 61.8 = ranging
    chopping = chop > 61.8  # True when ranging (avoid trading)
    trending = chop < 38.2  # True when trending (favor trading)
    
    # Session filter: 00-12 UTC (covers major session overlap)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 0) & (hours <= 12)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For Donchian20 and Chop14
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(chop[i]) or
            np.isnan(atr_1d[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Chop filter: avoid trading in ranging markets
        if chopping[i]:
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: ATR-based stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - 2.5 * atr_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= entry_price + 2.5 * atr_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume confirmation
            volume_confirm = volume[i] > (1.5 * vol_ma[i])
            
            long_breakout = (close[i] > donchian_high[i-1]) and volume_confirm
            short_breakout = (close[i] < donchian_low[i-1]) and volume_confirm
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals