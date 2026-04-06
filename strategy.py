#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume and ADX Filter
Hypothesis: Donchian channel breakouts capture trends, volume confirms participation,
and ADX filters out ranging markets. Works in bull (buy breakouts) and bear (sell breakdowns)
by using the same logic for both directions. Target: 80-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14357_4h_donchian_vol_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on daily data
    period = 14
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=period, min_periods=period).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=period, min_periods=period).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_high, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_low, min_periods=donchian_period).min().values
    
    # Volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.2 * vol_ma)  # Require 20% above average volume
    
    # ATR for stoploss (4h)
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.roll(close, 1))
    tr3_4h = np.abs(low - np.roll(close, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(donchian_period, 2*period) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_4h[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR ADX weak (< 20) OR stoploss
            if (close[i] <= donchian_low[i] or adx_aligned[i] < 20 or
                close[i] <= entry_price - 2.5 * atr_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR ADX weak (< 20) OR stoploss
            if (close[i] >= donchian_high[i] or adx_aligned[i] < 20 or
                close[i] >= entry_price + 2.5 * atr_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + strong trend (ADX > 25)
            long_breakout = close[i] > donchian_high[i]
            short_breakout = close[i] < donchian_low[i]
            
            if long_breakout and vol_filter[i] and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and vol_filter[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals