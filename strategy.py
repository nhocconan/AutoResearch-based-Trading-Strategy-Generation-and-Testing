#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_ATRStop_VolumeChop
Hypothesis: 4h Donchian(20) breakout with ATR-based stoploss, volume confirmation, and choppiness regime filter.
Uses Donchian channel breakouts as primary signal, filtered by volume > 1.5x average and choppy market regime (CHOP > 61.8).
ATR stoploss at 2.5x ATR below/above entry for longs/shorts. Designed for 75-200 total trades over 4 years (19-50/year).
Works in both bull and bear markets by adapting to volatility regimes and using volatility-based stops.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ATR(14) for volatility and stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index (CHOP) regime filter
    chop_window = 14
    sum_tr = pd.Series(tr).rolling(window=chop_window, min_periods=chop_window).sum().values
    highest_high_chop = pd.Series(high).rolling(window=chop_window, min_periods=chop_window).max().values
    lowest_low_chop = pd.Series(low).rolling(window=chop_window, min_periods=chop_window).min().values
    chop = 100 * np.log10(sum_tr / (highest_high_chop - lowest_low_chop) / np.log2(chop_window))
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Entry prices for stoploss calculation
    entry_price_long = np.full(n, np.nan)
    entry_price_short = np.full(n, np.nan)
    
    # Start after warmup
    start_idx = max(lookback, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(chop[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: Close breaks above Donchian upper + volume spike + choppy regime (CHOP > 61.8)
        if close[i] > highest_high[i] and volume_spike[i] and chop[i] > 61.8:
            if position != 1:
                signals[i] = base_size
                position = 1
                entry_price_long[i] = close[i]  # Record entry price for stoploss
            else:
                signals[i] = base_size
                entry_price_long[i] = entry_price_long[i-1] if i > 0 else close[i]
        # Short logic: Close breaks below Donchian lower + volume spike + choppy regime (CHOP > 61.8)
        elif close[i] < lowest_low[i] and volume_spike[i] and chop[i] > 61.8:
            if position != -1:
                signals[i] = -base_size
                position = -1
                entry_price_short[i] = close[i]  # Record entry price for stoploss
            else:
                signals[i] = -base_size
                entry_price_short[i] = entry_price_short[i-1] if i > 0 else close[i]
        # Exit logic: ATR-based stoploss or Donchian opposite breakout
        else:
            exit_signal = False
            if position == 1:
                # Long stoploss: price < entry - 2.5 * ATR
                if not np.isnan(entry_price_long[i-1]) and close[i] < entry_price_long[i-1] - 2.5 * atr[i]:
                    exit_signal = True
                # Alternative exit: close breaks below Donchian lower
                elif close[i] < lowest_low[i]:
                    exit_signal = True
            elif position == -1:
                # Short stoploss: price > entry + 2.5 * ATR
                if not np.isnan(entry_price_short[i-1]) and close[i] > entry_price_short[i-1] + 2.5 * atr[i]:
                    exit_signal = True
                # Alternative exit: close breaks above Donchian upper
                elif close[i] > highest_high[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price_long[i] = np.nan
                entry_price_short[i] = np.nan
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                    entry_price_long[i] = np.nan
                    entry_price_short[i] = np.nan
                elif position == 1:
                    signals[i] = base_size
                    entry_price_long[i] = entry_price_long[i-1] if i > 0 else np.nan
                    entry_price_short[i] = np.nan
                else:
                    signals[i] = -base_size
                    entry_price_long[i] = np.nan
                    entry_price_short[i] = entry_price_short[i-1] if i > 0 else np.nan
    
    return signals

name = "4h_Donchian20_Breakout_ATRStop_VolumeChop"
timeframe = "4h"
leverage = 1.0