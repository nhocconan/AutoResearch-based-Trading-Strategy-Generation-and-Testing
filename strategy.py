#!/usr/bin/env python3
"""
4h Donchian(20) breakout + 1d trend filter + volume confirmation + ATR stoploss
Hypothesis: Price breaking Donchian channels indicates institutional breakouts.
1d EMA200 filters trend to avoid counter-trend trades. Volume confirms breakout strength.
Works in bull (buy breakouts above) and bear (sell breakdowns below). Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_prev = np.roll(ema200_1d, 1)
    ema200_1d_prev[0] = ema200_1d[0]
    ema200_rising = ema200_1d > ema200_1d_prev
    ema200_falling = ema200_1d < ema200_1d_prev
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    ema200_rising_aligned = align_htf_to_ltf(prices, df_1d, ema200_rising)
    ema200_falling_aligned = align_htf_to_ltf(prices, df_1d, ema200_falling)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 200  # For EMA200
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(atr[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(ema200_rising_aligned[i]) or 
            np.isnan(ema200_falling_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse signal or stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR stoploss
            if (close[i] <= donchian_low[i] or 
                close[i] <= entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR stoploss
            if (close[i] >= donchian_high[i] or 
                close[i] >= entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend + volume
            bull_breakout = close[i] > donchian_high[i]
            bear_breakout = close[i] < donchian_low[i]
            
            if bull_breakout and ema200_rising_aligned[i] and volume[i] > vol_ema[i] * 1.5:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and ema200_falling_aligned[i] and volume[i] > vol_ema[i] * 1.5:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals