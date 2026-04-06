#!/usr/bin/env python3
"""
1h Range Reversal with 4h Trend Filter and Volume Confirmation
Hypothesis: In ranging markets (identified by low ADX on 4h), price reverses at 1h Bollinger Bands.
In trending markets (high ADX on 4d), trade pullbacks to EMA20. Volume confirms institutional participation.
Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend). Target: 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_range_reversal_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # 4h ADX for trend strength
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 4h EMA20 for pullback entries
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 4h indicators to 1h
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # 1h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For BBands and ADX
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(sma20[i]) or np.isnan(std20[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse signal or stoploss
        if position == 1:  # long position
            # Exit: price touches upper BB OR stoploss
            if (close[i] >= upper[i] or 
                close[i] <= entry_price - 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price touches lower BB OR stoploss
            if (close[i] <= lower[i] or 
                close[i] >= entry_price + 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: range or trend based on 4h ADX
            if adx_aligned[i] < 25:  # Ranging market
                # Mean reversion at Bollinger Bands
                long_entry = (close[i] <= lower[i] and 
                             volume[i] > vol_ema[i] * 1.5)
                short_entry = (close[i] >= upper[i] and 
                              volume[i] > vol_ema[i] * 1.5)
            else:  # Trending market
                # Pullback to EMA20
                long_entry = (close[i] <= ema20_4h_aligned[i] * 1.01 and  # within 1% of EMA
                             close[i] > ema20_4h_aligned[i] * 0.99 and
                             close[i] > close[i-1] and  # rising close
                             volume[i] > vol_ema[i] * 1.5)
                short_entry = (close[i] >= ema20_4h_aligned[i] * 0.99 and  # within 1% of EMA
                              close[i] < ema20_4h_aligned[i] * 1.01 and
                              close[i] < close[i-1] and  # falling close
                              volume[i] > vol_ema[i] * 1.5)
            
            if long_entry:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals