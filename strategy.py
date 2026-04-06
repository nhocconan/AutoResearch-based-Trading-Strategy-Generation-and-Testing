#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and Regime Filter
Hypothesis: Donchian(20) breakouts capture strong directional moves in both bull and bear markets.
Volume confirms institutional participation. Choppiness regime filter avoids whipsaws in sideways markets.
HTF 1d trend filters out counter-trend breakouts. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_vol_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)  # Require high volume
    
    # Choppiness regime filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr / (max_hh - min_ll + 1e-10)) / np.log10(14)
    
    # Choppiness regimes: >61.8 = ranging, <38.2 = trending
    chop_filter = chop < 61.8  # Avoid strong ranging markets
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For Donchian20 and EMA50
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(atr[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR stoploss
            if (close[i] <= low_20[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR stoploss
            if (close[i] >= high_20[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout + volume + regime + trend filter
            long_breakout = close[i] > high_20[i]
            short_breakout = close[i] < low_20[i]
            
            # 1d trend filter: only take long in uptrend, short in downtrend
            uptrend = ema_50_1d_aligned[i] > close[i]  # Price above 1d EMA50 = uptrend
            downtrend = ema_50_1d_aligned[i] < close[i]  # Price below 1d EMA50 = downtrend
            
            if long_breakout and vol_filter[i] and chop_filter[i] and uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and vol_filter[i] and chop_filter[i] and downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals